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

class AlphaMCTS {
public:

	Config config;
	std::shared_ptr<Shadow_Chess_V1_Resnet> model;

	AlphaMCTS(Config config, std::shared_ptr<Shadow_Chess_V1_Resnet> model) :
		config(config),
		model(model)
	{
# ifdef SDW_USE_CUDA
		this->model->to(torch::kCUDA);
		this->model->to_cuda();
# endif
	}

	torch::Tensor search(chess::Board state) {

	}
};