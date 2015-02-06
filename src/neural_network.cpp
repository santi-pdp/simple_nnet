#include "neural_network.h"

void Net::get_results(vector<double> &results_values) const{
	results_values.clear();
	for(unsigned n=0;n<m_layers.back().size()-1;++n){
		results_values.push_back(m_layers.back()[n].get_output_value());
	}
}		

void Net::back_prop(const vector<double> &target_values){
	//calculate overall net error (RMS of output neuron errors)
	Layer &output_layer = m_layers.back();	
	m_error = 0.0; //acumulate overall net err
	
	for(unsigned n=0;n<output_layer.size();n++){
		double delta = target_values[n] - output_layer[n].get_output_value();
		m_error += delta * delta; //sum of squares of the error
	}
	m_error /= output_layer.size()-1; //average
	m_error = sqrt(m_error);
	//implement a recent average measurement
	m_recent_avg_error = (m_recent_avg_error * m_recent_avg_smoothing_factor + m_error)/
		(m_recent_avg_smoothing_factor + 1.0);
	//calculate output layer gradients
	for(unsigned n=0;n<output_layer.size();++n){
		output_layer[n].calc_output_gradients(target_values[n]);
	}	
	//calculate gradients on hidden layers
	for(unsigned layer_num = m_layers.size()-2;layer_num>0;--layer_num){
		Layer &hidden_layer = m_layers[layer_num];
		Layer &next_layer = m_layers[layer_num+1];
		for(unsigned n=0;n<hidden_layer.size();++n){
			hidden_layer[n].calc_hidden_gradients(next_layer);
		}
	}
	//for all layers from outputs to first hidden layer, update connection weights
	for(unsigned layer_num=m_layers.size()-1;layer_num>0;--layer_num){
		Layer &layer = m_layers[layer_num];
		Layer &previous_layer = m_layers[layer_num -1];
		for(unsigned n=0;n<layer.size()-1;n++){
			layer[n].update_input_weights(previous_layer);
		}
	}
}

void Net::feed_forward(const vector<double> &input_values){
	assert(input_values.size() == m_layers[0].size()-1); //substract 1 because of the bias unit
	//Assign the input values to the input neurons
	for(unsigned i=0;i<input_values.size();++i){
		m_layers[0][i].set_output_value(input_values[i]);
	}
	//forward propagation (looping through each layer and then each neuron and tell each neuron to feed forward). Start from first hidden layer (1)
	for(unsigned layer_num=1;layer_num<m_layers.size();++layer_num){
		Layer &previous_layer = m_layers[layer_num-1];
		for(unsigned n=0;n<m_layers[layer_num].size()-1;++n){
			m_layers[layer_num][n].feed_forward(previous_layer);
		}
	}
}

Net::Net(const vector<unsigned> &topology){
	unsigned num_layers = topology.size();
	for (unsigned layer_num = 0; layer_num<num_layers; ++layer_num){
		m_layers.push_back(Layer());
		//take into account if we are in the output layer (0) or not
		unsigned num_outputs = layer_num == topology.size()-1? 0:topology[layer_num+1];
		//Now we need to add the neurons inside the layer
		for (unsigned neuron_num = 0; neuron_num <= topology[layer_num]; ++neuron_num){
			//m_layers.back() ens dona l'ultima capa construida
			m_layers.back().push_back(Neuron(num_outputs,neuron_num));	
			cout << "Made a neuron in layer (" << layer_num << ") and position (" << neuron_num << ")" << endl;
		}
		// force the bias node's output value to 1.0, it's the last neuron created above
		m_layers.back().back().set_output_value(1.0);
	}	
}
