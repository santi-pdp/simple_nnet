#include "training_data.h"

using namespace std;

/* GET TOPOLOGY FOR THE NETWORK
*********************************/
void training_data::get_topology(vector<unsigned> &topology){
	string line;
	string label;

	getline(m_training_data_file, line);
	stringstream ss(line);
	ss >> label;
	if(this->is_eof() || label.compare("topology:") != 0){
		return;
	}

	while(!ss.eof()){
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

/* GET NEXT INPUTS FROM THE TRAINING SET 
*****************************************/
unsigned training_data::get_next_inputs(vector<double> &input_values){
	input_values.clear();
	string line;
	getline(m_training_data_file, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if(label.compare("in:") == 0){
		double one_value;
		while(ss >> one_value){
			input_values.push_back(one_value);
		}
	}
	return input_values.size();
}

/* GET NEXT OUTPUTS FROM THE TRAINING SET 
*****************************************/
unsigned training_data::get_target_outputs(vector<double> &target_output_values){
	target_output_values.clear();
	string line;
	getline(m_training_data_file, line);
	stringstream ss(line);

	string label;
	ss >> label;
	if(label.compare("out:") == 0){
		double one_value;
		while( ss >> one_value) {
			target_output_values.push_back(one_value);
		}
	}
	return target_output_values.size();
}


/* CONSTRUCTOR
**************/
training_data::training_data(const string filename){
	m_training_data_file.open(filename.c_str());
}
