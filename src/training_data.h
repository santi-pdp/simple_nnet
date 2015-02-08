#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class training_data{
	public:
		/// training_data constructor	
		/// @param filename of the training data file
		training_data(const string filename);
		/// is_eof checks whether we've reached the end of the input data file
		/// @return true if we've reached the end of the file, false otherwise
		bool is_eof(void){ return m_training_data_file.eof(); }
		/// get_topology gets the topology information from the training_data file
		/// @param topology referenced vector to get the topology of the network
		void get_topology(vector<unsigned> &topology);

		/// get_next_inputs returns the number of input values from the file:
		/// @param input_values referenced vector to get the training input data
		/// @return number of input values
		unsigned get_next_inputs(vector<double> &input_values);
		/// get_target_outputs gets the output labels from the training data
		/// @param target_output_values referenced vector to get the training label data
		/// return number of output values
		unsigned get_target_outputs(vector<double> &target_output_values);

	private:
		ifstream m_training_data_file;
};


