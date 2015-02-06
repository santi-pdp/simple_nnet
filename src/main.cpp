#include "training_data.h"
#include "neural_network.h"

void show_vector_values(string label, vector<double> &v){
	cout << label << " ";
	for(unsigned i=0;i<v.size();++i){
		cout << v[i] << " ";
	}
	cout << endl;
}

int main(){
	// Topology e.g. (3, 2, 1)
	training_data train_data("training_data.txt");

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
	cout << endl << "Done with training!" << endl;
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
