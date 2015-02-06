#include <vector>
#include "training_data.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cassert>

using namespace std;

struct Connection{
	double weight;
	double delta_weight;
};

class Neuron;

typedef vector<Neuron> Layer;


// ******************** class Neuron **********************

class Neuron{
	public:
		Neuron(unsigned num_outputs,unsigned my_index);
		void set_output_value(double value){ m_output_value = value; }
		double get_output_value() const{ return m_output_value; }
		void feed_forward(const Layer &previous_layer);
		void calc_output_gradients(double target_value);
		void calc_hidden_gradients(const Layer &next_layer);
		void update_input_weights(Layer &previous_layer);
	private:
		static double eta; //{0.0...1.0} overall net training rate
		static double alpha; //{0.0...n} multiplier of the last weight change (momentum)
		static double transfer_function(double x);
		static double transfer_function_derivative(double x);
		static double random_weight(void){ return rand()/double(RAND_MAX); }
		double sum_DOW(const Layer &next_layer) const;
		double m_output_value;
		vector<Connection> m_output_weights;
		unsigned m_my_index;
		double m_gradient;
};

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

// ******************** class Net **********************

class Net{
	public:
		Net(const vector<unsigned> &topology);
		//reference to input_values that won't be modified
		void feed_forward(const vector<double> &input_values);
		void back_prop(const vector<double> &target_values);
		void get_results(vector<double> &results_values) const;
		double get_recent_avg_error(void){ return m_recent_avg_error; }
	private:
		vector<Layer> m_layers;	// m_layers[layerNum][neuronNum]
		double m_error;
		double m_recent_avg_error;
		double m_recent_avg_smoothing_factor;
};

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

void show_vector_values(string label, vector<double> &v){
	cout << label << " ";
	for(unsigned i=0;i<v.size();++i){
		cout << v[i] << " ";
	}
	cout << endl;
}

int main(){
	// Topology e.g. (3, 2, 1)
	training_data train_data("/tmp/training_data.txt");

	vector<unsigned> topology;
	train_data.get_topology(topology);
	Net my_net(topology);

	//To train the NN we call feed_forward
	vector<double> input_values, target_values, result_values;
	int training_pass = 0;
	while(!train_data.is_eof()){
		++training_pass;
		cout << endl << "Pass " << training_pass;
		//Get new input data and feed it forward
		if(train_data.get_next_inputs(input_values) != topology[0]){
			break;
		}
		show_vector_values(": Inputs:", input_values);
		my_net.feed_forward(input_values);

		//collect net's actual results
		my_net.get_results(result_values);
		show_vector_values("Outputs:", result_values);

		//Train the net what the outputs should have been
		train_data.get_target_outputs(target_values);
		show_vector_values("Targets:", target_values);
		assert(target_values.size() == topology.back());

		my_net.back_prop(target_values);
		//repot how well the training is working
		cout << "Net recent average error: " << my_net.get_recent_avg_error() << endl;
	}
	cout << endl << "Done" << endl;
	double arg1, arg2;
	while(true){
		cout << "insert first input: ";
		cin >> arg1;
		cout << endl << "insert second input: ";
		cin >> arg2;
		cout << endl;
		input_values.clear();
		input_values.push_back(arg1);
		input_values.push_back(arg2);
		my_net.feed_forward(input_values);
		my_net.get_results(result_values);
		show_vector_values("Outputs: ", result_values);
	}
	return 0;
}
