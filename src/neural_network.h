#include "neuron.h"


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

