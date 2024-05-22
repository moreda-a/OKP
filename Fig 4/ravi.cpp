#include <iostream>
#include <limits>
#include <cfloat>
#include <unordered_map>
#include <vector>
#include <queue>
#include <map>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <random>
#include <string>

using namespace std;

double capacity = 1000;
int minVal = 7;
int maxVal = 200;


struct KnapsackStat {
	double usedCap = 0;
	double profit = 0;
	int iStar = minVal - 1;
};

struct Info {
	double beta = 0;
	double profit = 0;
	double usedCap = 0;
	bool isValid = true;
};

Info findBeta(priority_queue <pair <int, int>> valCnt, int index, double alpha, vector <int>& lowerBounds, vector <int>& upperBounds) {
	double beta = 0.5;
	double delta = 0.25;
	double epsilon = 0.001;
	double sum2 = 0;

	for (int j = index + 1; j <= maxVal; j++) {
		sum2 += j * lowerBounds[j];
	}

	Info newInfo;
	while (!valCnt.empty() && valCnt.top().first > index && newInfo.usedCap < capacity) {
		int count = min(valCnt.top().second, (int)(capacity - newInfo.usedCap));
		newInfo.usedCap += count;
		newInfo.profit += count * valCnt.top().first;
		valCnt.pop();
	}

	if (newInfo.usedCap == capacity) {
		newInfo.isValid = false;
		return newInfo;
	}
	while (delta > 0) {
		double usedCap = newInfo.usedCap;
		double profit = newInfo.profit;
		//  ((1−β)u+βl elements of index
		double temp = (1 - beta) * (double)upperBounds[index] + beta * (double)lowerBounds[index];
		double indexCount = min(temp, capacity - usedCap);
		usedCap += indexCount;
		profit += indexCount * (double)index;

		priority_queue <pair <int, int>> restCnt = valCnt;

		while (!restCnt.empty() && usedCap < capacity) {
			int count = min(restCnt.top().second, (int)(capacity - usedCap));
			usedCap += count;
			profit += count * restCnt.top().first;
			restCnt.pop();
		}

		double sum1 = alpha * profit;
		if (abs(sum1 - sum2 - beta * (double)index * (double)lowerBounds[index]) < epsilon) {
			newInfo.beta = beta;
			newInfo.profit = profit;
			newInfo.usedCap = usedCap;
			return newInfo;
		}
		if (sum1 < sum2 + beta * (double)index * (double)lowerBounds[index]) {
			beta -= delta;
		}
		else {
			beta += delta;
		}
		delta /= 2;
	}
	return newInfo;
}

priority_queue <pair <int, int>> setMprime(int index, vector <int>& lowerBounds, vector <int>& upperBounds) {
	priority_queue <pair <int, int>> valCnt;
	if (index == minVal - 1) {
		for (int i = minVal + 1; i <= maxVal; i++) {
			if (lowerBounds[i] != 0) {
				valCnt.push({ i, lowerBounds[i] });
			}
		}
		return valCnt;
	}

	for (int i = minVal; i <= index; i++) {
		if (upperBounds[i] != 0) {
			valCnt.push({ i, upperBounds[i] });
		}
	}
	//skip index+1, will use enough of such values in findBeta
	for (int i = index + 2; i <= maxVal; i++) {
		if (lowerBounds[i] != 0) {
			valCnt.push({ i, lowerBounds[i] });
		}
	}
	return valCnt;
}

priority_queue <pair <int, int>> setM(int index, vector <int>& lowerBounds, vector <int>& upperBounds) {
	priority_queue <pair <int, int>> valCnt;
	if (index == minVal - 1) {
		for (int i = minVal; i <= maxVal; i++) {
			if (lowerBounds[i] != 0)
				valCnt.push({ i, lowerBounds[i] });
		}
		return valCnt;
	}

	for (int i = minVal; i <= index; i++) {
		if (upperBounds[i] != 0)
			valCnt.push({ i, upperBounds[i] });
	}

	for (int i = index + 1; i <= maxVal; i++) {
		if (lowerBounds[i] != 0)
			valCnt.push({ i, lowerBounds[i] });
	}
	return valCnt;
}

KnapsackStat findOpt(priority_queue <pair <int, int>> valCnt) {
	KnapsackStat newStat;
	double profit = 0;
	double usedCap = 0;
	while (!valCnt.empty() && usedCap < capacity) {
		double count = min(capacity - usedCap, (double)valCnt.top().second);
		usedCap += count;
		profit += ((double)(valCnt.top().first)) * count;
		valCnt.pop();
	}
	newStat.profit = profit;
	newStat.usedCap = usedCap;
	return newStat;
}


void findLowestValToPick(int& lowestVal, Info& newInfo, double alpha, vector <int>& lowerBounds, vector <int>& upperBounds) {
	lowestVal = minVal;
	for (int i = minVal; i <= maxVal; i++) {
		double head1 = alpha * findOpt(setM(i - 1, lowerBounds, upperBounds)).profit;
		double tail1 = alpha * findOpt(setM(i, lowerBounds, upperBounds)).profit;

		double head2 = 0;
		double tail2 = 0;
		for (int j = i + 1; j <= maxVal; j++) {
			head2 += j * lowerBounds[j];
		}
		tail2 = head2 + i * lowerBounds[i];
		if (max(head1, head2) > min(tail1, tail2)) {
			continue;
		}

		//ranges overlap
		lowestVal = i;
		priority_queue <pair <int, int>> Mprime = setMprime(i - 1, lowerBounds, upperBounds);
		newInfo = findBeta(Mprime, i, alpha, lowerBounds, upperBounds);
		return;
	}
}

pair <KnapsackStat, Info> phase1(double alpha, vector <int>& lowerBounds, vector <int>& upperBounds) {
	Info optMprimeInfo;
	int lowestVal;
	findLowestValToPick(lowestVal, optMprimeInfo, alpha, lowerBounds, upperBounds);

	KnapsackStat newStat;
	newStat.iStar = lowestVal;

	double usedCap = ceil(optMprimeInfo.beta * lowerBounds[lowestVal]);
	double profit = ceil(optMprimeInfo.beta * lowerBounds[lowestVal]) * lowestVal;
	for (int i = lowestVal + 1; i <= maxVal; i++) {
		if (usedCap >= capacity) {
			break;
		}
		int count = min((double)lowerBounds[i], capacity - usedCap);
		usedCap += count;
		profit += count * i;
	}
	newStat.profit = profit;
	newStat.usedCap = usedCap;
	return { newStat,optMprimeInfo };
}


vector <double> phase2(KnapsackStat currStat, Info optMprimeInfo, bool roundUp, double alpha, vector <int>& lowerBounds, vector <int>& upperBounds) {
	//set thresholds
	unordered_map <int, double> optM;
	vector <double> thresholds(maxVal + 1, 0);

	thresholds[minVal - 1] = currStat.usedCap;
	if (currStat.usedCap >= capacity)
		return thresholds;
	optM[currStat.iStar] = findOpt(setM(currStat.iStar, lowerBounds, upperBounds)).profit;

	thresholds[currStat.iStar] = (alpha * (optM[currStat.iStar] - optMprimeInfo.profit)) / (double)currStat.iStar;

	for (int i = currStat.iStar + 1; i <= maxVal; i++) {
		optM[i] = optM.count(i) == 0 ? findOpt(setM(i, lowerBounds, upperBounds)).profit : optM[i];
		optM[i - 1] = optM.count(i - 1) == 0 ? findOpt(setM(i - 1, lowerBounds, upperBounds)).profit : optM[i - 1];
		thresholds[i] = (alpha * (optM[i] - optM[i - 1])) / (double)i;
	}
	if (!roundUp)
		return thresholds;

	//round up the thresholds
	double sum = currStat.usedCap;
	for (int i = minVal; i <= maxVal; i++) {
		if (thresholds[i] == 0) {
			continue;
		}
		double temp = thresholds[i];
		thresholds[i] = ceil(temp + sum) - ceil(sum);
		sum += temp;
	}

	return thresholds;
}

KnapsackStat phase3(vector <double>& items, vector <double>& thresholds, KnapsackStat& currStat, int startIndex) {
	map <int, double> thresholdsMap;
	for (int i = minVal; i <= maxVal; i++) {
		if (thresholds[i] != 0) {
			thresholdsMap[i] = thresholds[i];
		}
	}

	for (int i = startIndex; i < items.size(); i++) {
		if (currStat.usedCap >= capacity || thresholdsMap.empty())
			break;
		auto it = thresholdsMap.lower_bound(items[i]);
		if (it == thresholdsMap.end())
			continue;
		if (it->first != items[i] && it == thresholdsMap.begin())
			continue;
		it = it->first == items[i] ? it : next(it, -1);
		currStat.usedCap++;
		currStat.profit += items[i];
		it->second--;
		if (it->second == 0) {
			thresholdsMap.erase(it);
		}
	}
	return currStat;
}


KnapsackStat ourAlg(vector <double>& onlineItems, vector <double>& thresholds, bool hasPhase3, bool roundUpInPhase2, int Phase3StartIndex, double alpha, vector <int>& lowerBounds, vector <int>& upperBounds) {

	pair <KnapsackStat, Info> phase1Res = phase1(alpha, lowerBounds, upperBounds);
	KnapsackStat currStat = phase1Res.first;
	Info optMprimeInfo = phase1Res.second;
	thresholds = phase2(currStat, optMprimeInfo, roundUpInPhase2, alpha, lowerBounds, upperBounds);
	if (!hasPhase3)
		return currStat;
	phase3(onlineItems, thresholds, currStat, Phase3StartIndex);
	return currStat;
}

double setAlpha(vector <int>& lowerBounds, vector <int>& upperBounds) {
	double alpha = 0.5;
	double delta = 0.25;
	double epsilon = 0.001;

	while (delta > 0.01) {
		vector <double> thresholds;
		vector <double> junk;
		ourAlg(junk, thresholds, false, false, -1, alpha, lowerBounds, upperBounds);
		double thresholdSum = 0;
		for (int i = minVal - 1; i <= maxVal; i++) {
			thresholdSum += thresholds[i];
		}
		if (abs(thresholdSum - capacity) <= epsilon)
			return alpha;

		if (thresholdSum < capacity)
			alpha += delta;
		else
			alpha -= delta;
		delta /= 2.0;
	}
	return alpha;
}

double psi(double filledFraction) {
	return pow((maxVal * exp(1) / minVal), filledFraction) * (minVal / exp(1));
}

KnapsackStat deeparnab(vector <double>& onlineItems) {
	double usedCap = 0;
	double profit = 0;

	for (int i = 0; i < onlineItems.size(); i++) {
		if (usedCap == capacity)
			break;
		if (onlineItems[i] >= psi(usedCap / capacity)) {
			usedCap++;
			profit += onlineItems[i];
		}
	}

	KnapsackStat currStat;
	currStat.profit = profit;
	currStat.usedCap = usedCap;
	return currStat;
}

void setOnlineOrder(vector <double>& onlineItems, vector <int> lowerBounds, int sigmaL) {
	sigmaL = min(sigmaL, (int)onlineItems.size());
	int index = -1;
	for (int i = 0; i < onlineItems.size(); i++) {
		if (index == sigmaL - 1) {
			break;
		}
		if (lowerBounds[onlineItems[i]] > 0) {
			lowerBounds[onlineItems[i]]--;
			swap(onlineItems[i], onlineItems[++index]);
		}
	}

	sort(onlineItems.begin(), onlineItems.begin() + sigmaL);
}

struct averageRes {
	map <double, double> DeeparnabProfits;
	map <double, double> DeeparnabUsedCap;

	map <double, double> ourProfits;
	map <double, double> ourUsedCap;

	map <double, double> optProfits;
	map <double, double> optUsedCap;

	map <double, double> geometricAlpha;
	averageRes(double maxDelta, double deltaSteps) {
		for (double delta = 0; delta <= maxDelta; delta += deltaSteps) {
			DeeparnabProfits[delta] = 1;
			DeeparnabUsedCap[delta] = 1;
			ourProfits[delta] = 1;
			ourUsedCap[delta] = 1;
			optProfits[delta] = 1;
			optUsedCap[delta] = 1;
			geometricAlpha[delta] = 1;
		}
	}
};

void generateInput(double delta, int sigmaL, vector <int> lowerBounds, vector <int> upperBounds, averageRes& finalRes, vector<double>& numbers, int repeat) {
	double alpha = setAlpha(lowerBounds, upperBounds);
	finalRes.geometricAlpha[delta] *= alpha;
	cout << "y\n";
	priority_queue <pair <int, int>> valueCount;

	map<int, int> mp;


	// l item of each value arrives first
	for (double num : numbers) {
		if (!mp.count(num))
			mp[num] = 0;
		mp[num]++;
	}


	// the rest of the items arrive
	for (int i = minVal; i <= maxVal; i++) {
		if (mp.count(i))
			valueCount.push({ i,mp[i] });
	}

	//random_shuffle(numbers.begin(), numbers.end());
	//KnapsackStat deeparnabRes = deeparnab(numbers);
	//finalRes.DeeparnabProfits[delta] *= deeparnabRes.profit;
	//finalRes.DeeparnabUsedCap[delta] *= deeparnabRes.usedCap;

	//KnapsackStat offlineRes = findOpt(valueCount);
	//finalRes.optProfits[delta] *= offlineRes.profit;
	//finalRes.optUsedCap[delta] *= offlineRes.usedCap;

	vector <double> thresholds;
	setOnlineOrder(numbers, lowerBounds, sigmaL);
	KnapsackStat currStat = ourAlg(numbers, thresholds, true, true, sigmaL, alpha, lowerBounds, upperBounds);
	finalRes.ourProfits[delta] *= currStat.profit;
	finalRes.ourUsedCap[delta] *= currStat.usedCap;
}
vector<double> readNumbersFromFile(const string& filename) {
	ifstream inFile(filename);
	vector<double> numbers;

	if (!inFile) {
		cerr << "Failed to open the file." << endl;
		return numbers; // Return empty vector if file can't be opened
	}

	double number;
	while (inFile >> number) {
		numbers.push_back((int)number / 10); // Store number in vector
	}

	inFile.close(); // Close the file
	return numbers;
}
void setDelta(vector<double>& numbers) {
	double maxDelta = 2.03;
	double deltaSteps = 5.5;
	averageRes finalRes(maxDelta, deltaSteps);
	int repeatCount = 5;
	for (int repeat = 0; repeat < repeatCount; repeat++) {
		//numbers = readNumbersFromFile("random_numbers" + to_string(repeat + 1) + ".txt");
		int k = maxVal;
		vector <int> lowerBounds(k + 1, 0);
		vector <int> upperBounds(k + 1, 0);

		int llowerbound = 0;
		int lupperbound = 3;
		int sigmaL = 0;
		for (int i = minVal; i <= maxVal; i++) {
			int lowerbound = rand() % (lupperbound - llowerbound + 1) + llowerbound;
			lowerBounds[i] = 0;
			sigmaL += lowerBounds[i];
		}

		for (double delta = 0; delta <= maxDelta; delta += deltaSteps) {
			for (int i = minVal; i <= maxVal; i++) {
				//upperBounds[i] = ceil(((double)lowerBounds[i]) * (1.0 + delta));
				upperBounds[i] = ceil((double)10 * numbers.size() / (maxVal - minVal));
			}
			generateInput(delta, sigmaL, lowerBounds, upperBounds, finalRes, numbers, repeat);
			cout << "x\n";
		}
	}

	cout << "our competitive ratio: " << endl;
	for (auto it = finalRes.ourProfits.begin(); it != finalRes.ourProfits.end(); it++) {
		cout << finalRes.optProfits[it->first] / it->second << " , ";//<< " :X: " << finalRes.optProfits[it->first] << " :Y: " << it->second << " , ";
	}
	cout << endl;


	cout << "Deeparnab's competitive ratio: " << endl;
	for (auto it = finalRes.DeeparnabProfits.begin(); it != finalRes.DeeparnabProfits.end(); it++) {
		cout << it->second / finalRes.optProfits[it->first] << ", ";
	}
	cout << endl;

	cout << "our theoretical competitive ratio" << endl;
	for (auto it = finalRes.geometricAlpha.begin(); it != finalRes.geometricAlpha.end(); it++) {
		it->second = pow(it->second, 1.0 / (double)repeatCount);
		cout << it->second << ", ";
	}
	cout << endl;

	cout << "Deeparnab's theoretical competitive ratio: " << endl;
	double ln = log((double)maxVal / (double)minVal) / log(exp(1));
	cout << 1.0 / (ln + 1.0) << endl;

}


void writeRandomNumbers(int k, int x, int y, const string& filename) {
	random_device rd;  // Obtain a random number from hardware
	mt19937 gen(rd()); // Seed the generator
	uniform_int_distribution<> distr(x, y); // Define the range
	for (int j = 1; j <= 12;++j) {
		ofstream outFile(filename + "_" + to_string(j) + ".txt");
		if (!outFile) {
			cerr << "Failed to open the file." << endl;
			return;
		}

		for (int i = 0; i < k; ++i) {
			outFile << distr(gen) << endl; // Generate and write a random number
		}

		outFile.close(); // Close the file
	}
}
double generateRandomNumber(double x, double y) {

	// Create a random device to seed the random number generator
	random_device rd;
	// Use the Mersenne Twister engine for random number generation
	mt19937 gen(rd());
	// Define the distribution range [x, y]
	uniform_real_distribution<> dis(x, y);

	// Generate and return the random number
	return dis(gen);
}

double setDelta2(vector<double>& numbers, double dd) {
	double maxDelta = 2.03;
	double deltaSteps = 5.5;
	averageRes finalRes(maxDelta, deltaSteps);
	int repeatCount = 1;
	srand(time(nullptr));
	for (int repeat = 0; repeat < repeatCount; repeat++) {
		//numbers = readNumbersFromFile("random_numbers" + to_string(repeat + 1) + ".txt");
		int k = maxVal;
		vector <int> lowerBounds(k + 1, 0);
		vector <int> upperBounds(k + 1, 0);


		priority_queue <pair <int, int>> valueCount;

		map<int, int> mp;


		// l item of each value arrives first
		for (double num : numbers) {
			if (!mp.count(num))
				mp[num] = 0;
			mp[num]++;
		}

		int sigmaL = 0;

		// the rest of the items arrive
		int lowitem = -1;
		int highitem = -1;
		for (int i = minVal; i <= maxVal; i++) {
			/*if (mp.count(i))
			{
				if (lowitem == -1)
					lowitem = i;
				double f = generateRandomNumber(1, 1 + dd);

				lowerBounds[i] = min((int)max(0, (int)floor(mp[i] / f)), (int)mp[i]);
				upperBounds[i] = max((int)floor(mp[i] * f), (int)mp[i] + 5);
				highitem = i;
			}
			else
			{
				lowerBounds[i] = 0;upperBounds[i] = 0;
			}*/
			lowerBounds[i] = 10;
			upperBounds[i] = 90;
			sigmaL += lowerBounds[i];
		}
		/*for (int i = lowitem; i <= highitem; i++) {
			if (mp.count(i))
			{

			}
			else
			{
				lowerBounds[i] = 0;upperBounds[i] = 10;
			}
		}*/

		//
		// for (double delta = 0; delta <= maxDelta; delta += deltaSteps) {
		{

			generateInput(0, sigmaL, lowerBounds, upperBounds, finalRes, numbers, repeat);
			cout << "x\n";
		}
	}

	cout << "our competitive ratio: " << endl;
	for (auto it = finalRes.ourProfits.begin(); it != finalRes.ourProfits.end(); it++) {
		//cout << finalRes.optProfits[it->first] / it->second << " , ";//<< " :X: " << finalRes.optProfits[it->first] << " :Y: " << it->second << " , ";
		cout << it->second / 10;
		return it->second / 10;
	}
	//cout << endl;


	//cout << "Deeparnab's competitive ratio: " << endl;
	//for (auto it = finalRes.DeeparnabProfits.begin(); it != finalRes.DeeparnabProfits.end(); it++) {
	//	cout << it->second / finalRes.optProfits[it->first] << ", ";
	//}
	//cout << endl;

	//cout << "our theoretical competitive ratio" << endl;
	//for (auto it = finalRes.geometricAlpha.begin(); it != finalRes.geometricAlpha.end(); it++) {
	//	it->second = pow(it->second, 1.0 / (double)repeatCount);
	//	cout << it->second << ", ";
	//}
	//cout << endl;

	//cout << "Deeparnab's theoretical competitive ratio: " << endl;
	//double ln = log((double)maxVal / (double)minVal) / log(exp(1));
	//cout << 1.0 / (ln + 1.0) << endl;

}

double processFile(int fileNumber, double delta) {
	vector<double> numbers;

	string filename = "rand_numbers_" + to_string(fileNumber) + ".txt";
	ifstream infile(filename);
	double result = 0.0;

	if (infile.is_open()) {
		double num;
		while (infile >> num) {
			// Process each number based on delta (example: add delta)
			numbers.push_back((int)num / 100);
		}
		infile.close();
	}
	else {
		cerr << "Unable to open file: " << filename << endl;
	}

	return setDelta2(numbers, delta);
}
// Function to write the output to a file
void writeOutputToFile(const  string& outputFilename) {
	ofstream outfile(outputFilename);
	if (outfile.is_open()) {
		// Specify the delta values to check
		double deltas[] = { 0.0, 0.1, 0.2, 0.3, 0.4 };
		int num_deltas = sizeof(deltas) / sizeof(deltas[0]);

		// Loop over delta values
		for (int i = 0; i < num_deltas; ++i) {
			double delta = deltas[i];
			// Loop over file numbers
			for (int j = 1; j <= 12; ++j) {
				double result = processFile(j, delta);
				outfile << fixed << setprecision(2) << delta << " " << j << " " << result << endl;
			}
		}
		outfile.close();
		cout << "Output written to " << outputFilename << endl;
	}
	else {
		cerr << "Unable to open file: " << outputFilename << endl;
	}
}


// Function to write vector elements to a file with a specific naming pattern
void writeVectorToFile(const std::vector<double>& vec, int repeat, double delta) {
	// Construct the file name
	std::string fileName = "ravi_" + std::to_string(repeat) + "_" + std::to_string(static_cast<int>(delta * 10)) + ".txt";

	// Open the file for writing
	std::ofstream outFile(fileName);

	// Check if the file was opened successfully
	if (!outFile) {
		std::cerr << "Error opening file: " << fileName << std::endl;
		return;
	}

	// Write the first line with the number of elements
	outFile << vec.size() << std::endl;

	// Write each element of the vector on a new line
	for (auto element : vec) {
		outFile << static_cast<int>(element) << std::endl;
	}

	// Close the file
	outFile.close();
}

double generateIn(double delta, int sigmaL, vector <int> lowerBounds, vector <int> upperBounds, averageRes& finalRes, int repeat) {
	double alpha = setAlpha(lowerBounds, upperBounds);
	finalRes.geometricAlpha[delta] *= alpha;

	priority_queue <pair <int, int>> valueCount;
	vector <double> onlineItems;
	int total = 0;

	// l item of each value arrives first
	for (int i = minVal; i <= maxVal; i++) {
		for (int j = 0; j < lowerBounds[i]; j++) {
			onlineItems.push_back(i);
		}
	}

	// the rest of the items arrive
	for (int i = minVal; i <= maxVal; i++) {
		int ni = lowerBounds[i] + rand() % (upperBounds[i] - lowerBounds[i] + 1);

		if (ni != 0) {
			valueCount.push({ i,ni });
		}
		total += ni;
		for (int j = 0; j < ni - lowerBounds[i]; j++) {
			onlineItems.push_back(i);
		}
	}

	random_shuffle(onlineItems.begin(), onlineItems.end());
	//write repeat and delta
	writeVectorToFile(onlineItems, repeat + 1, delta);
	/*KnapsackStat deeparnabRes = deeparnab(onlineItems);
	finalRes.DeeparnabProfits[delta] *= deeparnabRes.profit;
	finalRes.DeeparnabUsedCap[delta] *= deeparnabRes.usedCap;

	KnapsackStat offlineRes = findOpt(valueCount);
	finalRes.optProfits[delta] *= offlineRes.profit;
	finalRes.optUsedCap[delta] *= offlineRes.usedCap;*/

	vector <double> thresholds;
	setOnlineOrder(onlineItems, lowerBounds, sigmaL);
	KnapsackStat currStat = ourAlg(onlineItems, thresholds, true, true, sigmaL, alpha, lowerBounds, upperBounds);
	finalRes.ourProfits[delta] *= currStat.profit;
	finalRes.ourUsedCap[delta] *= currStat.usedCap;
	//if (currStat.usedCap > capacity)
	//	cout << "\nSHIT:" << currStat.usedCap;

	return currStat.profit;
}

void version1() {
	capacity = 10000;
	minVal = 1;
	maxVal = 100;


	//OPEN FILE 
	ofstream outt("im0.txt");
	ofstream outt2("adv0.txt");
	if (outt.is_open()) {
		//// Specify the delta values to check
		//double deltas[] = { 0.0, 0.1, 0.2, 0.3, 0.4 };
		//int num_deltas = sizeof(deltas) / sizeof(deltas[0]);

		//// Loop over delta values
		//for (int i = 0; i < num_deltas; ++i) {
		//	double delta = deltas[i];
		//	// Loop over file numbers
		//	for (int j = 1; j <= 12; ++j) {
		//		double result = processFile(j, delta);
		//	}
		//}



	//setDelta
		double maxDelta = 2;
		double deltaSteps = 0.5;
		averageRes finalRes(maxDelta, deltaSteps);
		int repeatCount = 10;
		for (int repeat = 0; repeat < repeatCount; repeat++) {
			int k = maxVal;
			vector <int> lowerBounds(k + 1, 0);
			vector <int> upperBounds(k + 1, 0);

			int llowerbound = 50;
			int lupperbound = 150;
			int sigmaL = 0;
			for (int i = minVal; i <= maxVal; i++) {
				int lowerbound = rand() % (lupperbound - llowerbound + 1) + llowerbound;
				lowerBounds[i] = lowerbound;
				sigmaL += lowerBounds[i];
			}

			for (double delta = 0; delta <= maxDelta; delta += deltaSteps) {
				double fill = 0;
				int adv = 0;
				for (int i = minVal; i <= maxVal; i++) {
					upperBounds[i] = ceil(((double)lowerBounds[i]) * (1.0 + delta));
				}
				for (int i = maxVal; i >= minVal;i--) {
					if (fill < capacity)
						fill += (upperBounds[i] + lowerBounds[i]) / 2;
					if (fill >= capacity) {
						adv = i;
						break;
					}
				}

				double result = generateIn(delta, sigmaL, lowerBounds, upperBounds, finalRes, repeat) / 10000;

				outt << fixed << setprecision(4) << delta << " " << repeat + 1 << " " << result << endl;
				outt2 << fixed << setprecision(4) << delta << " " << repeat + 1 << " " << adv << endl;
			}
		}


		outt.close();
		outt2.close();
		cout << "Output written to " << "im0.txt" << endl;
	}
	else {
		cerr << "Unable to open file: " << "im0.txt" << endl;
	}

}








double generateIn1(double delta, int sigmaL, vector <int> lowerBounds, vector <int> upperBounds, averageRes& finalRes, vector<double>& numbers, int repeat) {
	double alpha = setAlpha(lowerBounds, upperBounds);
	finalRes.geometricAlpha[delta] *= alpha;
	cout << "y\n";
	priority_queue <pair <int, int>> valueCount;

	map<int, int> mp;


	// l item of each value arrives first
	for (double num : numbers) {
		if (!mp.count(num))
			mp[num] = 0;
		mp[num]++;
	}


	// the rest of the items arrive
	for (int i = minVal; i <= maxVal; i++) {
		if (mp.count(i))
			valueCount.push({ i,mp[i] });
	}

	//random_shuffle(numbers.begin(), numbers.end());
	//KnapsackStat deeparnabRes = deeparnab(numbers);
	//finalRes.DeeparnabProfits[delta] *= deeparnabRes.profit;
	//finalRes.DeeparnabUsedCap[delta] *= deeparnabRes.usedCap;

	//KnapsackStat offlineRes = findOpt(valueCount);
	//finalRes.optProfits[delta] *= offlineRes.profit;
	//finalRes.optUsedCap[delta] *= offlineRes.usedCap;

	vector <double> thresholds;
	setOnlineOrder(numbers, lowerBounds, sigmaL);
	KnapsackStat currStat = ourAlg(numbers, thresholds, true, true, sigmaL, alpha, lowerBounds, upperBounds);
	finalRes.ourProfits[delta] *= currStat.profit;
	finalRes.ourUsedCap[delta] *= currStat.usedCap;
	return currStat.profit;
}


void version2() {
	capacity = 1000;
	minVal = 70;
	maxVal = 2000;


	//OPEN FILE 
	ofstream outt("im1.txt");
	ofstream outt2("adv1.txt");
	if (outt.is_open()) {
		//// Specify the delta values to check
		//double deltas[] = { 0.0, 0.1, 0.2, 0.3, 0.4 };
		//int num_deltas = sizeof(deltas) / sizeof(deltas[0]);

		//// Loop over delta values
		//for (int i = 0; i < num_deltas; ++i) {
		//	double delta = deltas[i];
		//	// Loop over file numbers
		//	for (int j = 1; j <= 12; ++j) {
		//		double result = processFile(j, delta);
		//	}
		//}



	//setDelta
		double maxDelta = 2;
		double deltaSteps = 0.5;
		averageRes finalRes(maxDelta, deltaSteps);
		int repeatCount = 24;
		for (int repeat = 0; repeat < repeatCount; repeat++) {
			int k = maxVal;
			vector <int> lowerBounds(k + 1, 0);
			vector <int> upperBounds(k + 1, 0);

			//load data of the month

			priority_queue <pair <int, int>> valueCount;

			map<int, int> mp;
			string fileName = "con_numbers_" + std::to_string(repeat+1) +".txt";
			vector<double> numbers = readNumbersFromFile(fileName);

			// l item of each value arrives first
			for (double num : numbers) {
				if (!mp.count(num))
					mp[num] = 0;
				mp[num]++;
			}


			for (double delta = 0; delta <= maxDelta; delta += deltaSteps) {
				int sigmaL = 0;




				for (int i = minVal; i <= maxVal; i++) {
					if (mp.count(i))
					{
						double f = generateRandomNumber(1, 1 + delta);
						double g = generateRandomNumber(1, 1 + delta);

						lowerBounds[i] = min((int)max(0, (int)floor(mp[i] / f)), (int)mp[i]);
						upperBounds[i] = max((int)floor(mp[i] * g), (int)mp[i]);
						//highitem = i;
					}
					else
					{
						lowerBounds[i] = 0;upperBounds[i] = 0;
					}
					sigmaL += lowerBounds[i];
				}











				double fill = 0;
				int adv = 0;
				for (int i = minVal; i <= maxVal; i++) {
					upperBounds[i] = ceil(((double)lowerBounds[i]) * (1.0 + delta));
				}
				for (int i = maxVal; i >= minVal;i--) {
					if (fill < capacity)
						fill += (upperBounds[i] + lowerBounds[i]) / 2;
					if (fill >= capacity) {
						adv = i;
						break;
					}
				}

				double result = generateIn1(delta, sigmaL, lowerBounds, upperBounds, finalRes,numbers ,repeat) / 100;

				outt << fixed << setprecision(4) << delta << " " << repeat + 1 << " " << result << endl;
				outt2 << fixed << setprecision(4) << delta << " " << repeat + 1 << " " << adv*10 << endl;
			}
		}


		outt.close();
		outt2.close();
		cout << "Output written to " << "im1.txt" << endl;
	}
	else {
		cerr << "Unable to open file: " << "im1.txt" << endl;
	}

}












int main(int argc, const char* argv[]) {
	//writeRandomNumbers(10000, 700, 20000, "rand_numbers");
	////int n = capacity * 10; // Number of random numbers to generate
	////int x = minVal;  // Lower bound of the range
	////int y = maxVal; // Upper bound of the range
	////string filename = "random_numbers.txt"; // File to write the random numbers

	////// Generate random numbers and write them to a file
	//////
	//////cout << "Random numbers have been written to " << filename << endl;

	////// Read numbers from the file
	////vector<double> numbers = readNumbersFromFile(filename);
	////cout << "Reading numbers from " << filename << ":" << endl;

	////setDelta(numbers);

	//////// Print the numbers from the vector
	//////for (int num : numbers) {
	//////    cout << num << endl;
	////}

	//string outputFilename = "im2.txt"; // Specify the output filename

	//// Write the output to the file
	//writeOutputToFile(outputFilename);

	//version1();
	version2();

}