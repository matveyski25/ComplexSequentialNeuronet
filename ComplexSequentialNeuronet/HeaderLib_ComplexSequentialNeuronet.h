#pragma once

#include <iostream>

#include <fstream>
#include <stdexcept>
#include <string> 
#include <sstream>
#include <algorithm>

#include <Eigen/Dense>

#include <ActivateFunctionsForNN/HeaderActivateFunctionsForNN.h>

/*void save_vector(std::ofstream& file, const std::vector<MatrixXld>& vec) const {
	file << vec.size() << "\n";
	for (const auto& m : vec) {
		save_matrix(file, m);
	}
}

void load_vector(std::ifstream& file, std::vector<MatrixXld>& vec) {
	size_t size;
	file >> size;
	vec.resize(size);
	for (auto& m : vec) {
		load_matrix(file, m);
	}
}*/

