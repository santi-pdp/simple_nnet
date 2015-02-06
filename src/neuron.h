#include <vector>
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
