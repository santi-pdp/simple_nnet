#include "neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5; 

void Neuron::update_input_weights(Layer &previous_layer){
	//The weights to be updated are in the conn container in the neurons
	//in the preceding layer
	for(unsigned n=0;n<previous_layer.size();++n){
		Neuron &neuron = previous_layer[n];
		double old_delta_weight = neuron.m_output_weights[m_my_index].delta_weight;
		double new_delta_weight = eta *neuron.get_output_value()*m_gradient
			+ alpha*old_delta_weight;
		neuron.m_output_weights[m_my_index].delta_weight = new_delta_weight;
		neuron.m_output_weights[m_my_index].weight += new_delta_weight;
	}
}
		
double Neuron::sum_DOW(const Layer &next_layer) const{
	double sum = 0.0;
	//Sum our contributions of the errs at the nodes we feed
	for(unsigned n=0;n<next_layer.size()-1;++n){
		sum += m_output_weights[n].weight * next_layer[n].m_gradient;
	}
	return sum;
}

void Neuron::calc_hidden_gradients(const Layer &next_layer){
	double dow = sum_DOW(next_layer);
	m_gradient = dow * Neuron::transfer_function_derivative(m_output_value);
}

void Neuron::calc_output_gradients(double target_value){
	double delta = target_value - m_output_value;
	m_gradient = delta * Neuron::transfer_function_derivative(m_output_value);
}
double Neuron::transfer_function(double x){
	// tanh - output range {-1.0 ... 1.0}
	return tanh(x);
}

double Neuron::transfer_function_derivative(double x){
	// tanh derivative approximation
	return 1.0-x*x;
}

void Neuron::feed_forward(const Layer &previous_layer){
	double sum = 0.0;
	//Sum the prev layer's outputs
	//Include the bias node from prev layer
	for(unsigned n=0; n<previous_layer.size();++n){
		//sum up all contributions with the corresponding weights of previous neurons
		sum+=previous_layer[n].get_output_value()*
			previous_layer[n].m_output_weights[m_my_index].weight;
	}
	//apply the transfer function now
	m_output_value = Neuron::transfer_function(sum);
}

Neuron::Neuron(unsigned num_outputs, unsigned my_index){
	for (unsigned c = 0; c<num_outputs; ++c){
		m_output_weights.push_back(Connection());
		m_output_weights.back().weight = random_weight();
	}
	m_my_index = my_index;
}

