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

	bool is_expanded() {
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
	}

	float get_ucb(Node* child) {
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