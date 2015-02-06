#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class training_data{
	public:
		training_data(const string filename);
		bool is_eof(void){ return m_training_data_file.eof(); }
		void get_topology(vector<unsigned> &topology);

		//returns the number of input values from the file:
		unsigned get_next_inputs(vector<double> &input_values);
		unsigned get_target_outputs(vector<double> &target_output_values);

	private:
		ifstream m_training_data_file;
};


