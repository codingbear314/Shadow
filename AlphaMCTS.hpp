#pragma once

#include <vector>
#include <cmath>

#include "chess.hpp"
#include "Encode.hpp"
#include "ShadowModel.hpp"

struct Config {
	float C;
};

class Node {
public:
	Config config;
	chess::Board state;
	Node* parent;
	int parent_action;
	int prior;

	std::vector<Node*> children;

	int visit_count;
	float value_sum;

	Node(
		Config config,
		chess::Board state,
		Node* parent,
		int parent_action,
		int prior = 0,
		int visit_count = 0
	) :
		config(config),
		state(state),
		parent(parent),
		parent_action(parent_action),
		prior(prior),
		visit_count(visit_count),
		value_sum(0) 
	{}

	bool is_expanded() const {
		return children.size() > 0;
	}

	void select_leaf(Node* selected) {
		selected = NULL;
		float best_ucb = -std::numeric_limits<float>::infinity();

		for (int i = 0; i < children.size(); i++) {
			Node* child = children[i];
			float ucb = this->get_ucb(child);
			if (ucb > best_ucb) {
				best_ucb = ucb;
				selected = child;
			}
		}

		return;
	}

	void expand(torch::Tensor policy) { // Shape : [1, 4672]
		chess::Movelist mvl;
		chess::movegen::legalmoves(mvl, state);
		for (int i = 0; i < mvl.size(); i++) {
			chess::Move move = mvl[i];
			chess::Board new_state = state;
			new_state.makeMove(move);
			new_state.mirrorBoard();
			int int_action = EncodeMove(move);
			auto poli = policy[0][int_action].item<float>();
			if (poli <= 0)
				continue;
			Node* child = new Node(this->config, new_state, this, int_action, poli);
			children.push_back(child);
		}
	}

	void backpropagate(float value) {
		this->visit_count += 1;
		this->value_sum += value;
		value = -value;
		if (this->parent) {
			this->parent->backpropagate(value);
		}
	}

	float get_ucb(const Node* child) const {
		float q_value = 0;
		if (child->visit_count) {
			q_value = ((child->value_sum / child->visit_count) + 1.0)/(float(2));
		}
		return q_value +
			this->config.C *
			(sqrt(this->visit_count) / (1 + child->visit_count)) *
			child->prior;
	}
};

// Alpha MCTS, done with multithreading
class ParallelAlphaMCTS {
public:
	// Logic :
	/*
	* We make a input tensor with (parallel games, 1, 8, 8, 6) by multithreading.
	* The main thread looks out for when the parallel counter is done.
	* Then, we run the model on the input tensor.
	* We save the output tensor as a variable, and make the atomic counter 0 again.
	* Then, on the threads, they go on with the logic
	*/

	torch::Tensor input_tensor;
	torch::Tensor output_tensor_policy;
	torch::Tensor output_tensor_value;
	std::atomic<int> parallel_counter;
	std::atomic<int> ongoing_games;
	std::atomic<bool> ModelDone;
	std::vector<std::thread> threads;
	Node* roots[8];
	std::mutex inputWrite;
	Config config;

	ParallelAlphaMCTS(
		torch::Tensor input_tensor,
		torch::Tensor output_tensor,
		Config config
	) :
		input_tensor(input_tensor),
		output_tensor(output_tensor),
		parallel_counter(0),
		config(config)
	{}

	void threaded_iteration(Node *root) {
		// Not the first, so no need for dirichlet noise
		while (root->is_expanded()) {
			root->select_leaf(root);
		}

		// Check if the game is over
		float value = 0;
		auto over = root->state.isGameOver();
		if (over.second == chess::GameResult::DRAW) value = 0;
		else if (over.second == chess::GameResult::WIN) value = -1;
		else if (over.second == chess::GameResult::LOSE) value = 1;

		else {
			this->ModelDone = false;
			this->inputWrite.lock();
			int pointer = this->parallel_counter;
			this->input_tensor[parallel_counter][0] = encode_board(root->state);
			this->parallel_counter += 1;
			this->inputWrite.unlock();
			while (this->ModelDone == false) {}
			this->inputWrite.lock();
			torch::Tensor policy = this->output_tensor_policy[pointer];
			float value = this->output_tensor_value[pointer].item<float>();

			this->inputWrite.unlock();

		}
	}

	void iteration() {
		// Start the threads

	}
};