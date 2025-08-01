#include "HeaderSeq2seqWithAttention.h"
#include <unordered_map>
#include <windows.h>
#ifdef max
#undef max
#endif 

class Dictionary {
public:
    std::unordered_map<char, RowVectorXld> data_ch2vec;  // char -> vec
    std::vector<char> index2ch;                          // index -> char
    std::vector<RowVectorXld> all_vectors;               // index -> vec

    Eigen::Index emb_size = 0;

    void push(char ch_, const RowVectorXld& vec_) {
        if (emb_size == 0) emb_size = vec_.cols();
        else if (vec_.cols() != emb_size)
            throw std::invalid_argument("–азмер вектора не совпадает с emb_size");

        if (data_ch2vec.count(ch_) == 0) {
            index2ch.push_back(ch_);
            all_vectors.push_back(vec_);
        }
        else {
            int idx = char_to_index(ch_);
            all_vectors[idx] = vec_;
        }
        data_ch2vec[ch_] = vec_;
    }

    RowVectorXld operator[](char ch_) const {
        if (!data_ch2vec.count(ch_))
            throw std::out_of_range("—имвол не найден в словаре");
        return data_ch2vec.at(ch_);
    }

    
    MatrixXld getEmbedding(const std::string& word_) const {
        MatrixXld emb(word_.size(), emb_size);
        for (Eigen::Index i = 0; i < word_.size(); ++i) {
            auto it = data_ch2vec.find(word_[i]);
            if (it != data_ch2vec.end())
                emb.row(i) = it->second;
            else
                emb.row(i) = RowVectorXld::Zero(emb_size);  // паддинг
        }
        return emb;
    }

   
    std::string getWords(const MatrixXld& mat) const {
        std::string result;
        for (Eigen::Index i = 0; i < mat.rows(); ++i) {
            result += (*this)[mat.row(i)];  // ближайший символ
        }
        return result;
    }

    // ќбратное отображение: вектор -> ближайший символ
    char operator[](const RowVectorXld& vec_) const {
        double min_dist = std::numeric_limits<double>::max();
        int best_idx = -1;
        for (size_t i = 0; i < all_vectors.size(); ++i) {
            double dist = (all_vectors[i] - vec_).squaredNorm();
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = static_cast<int>(i);
            }
        }
        return index2ch[best_idx];
    }

    std::vector<MatrixXld> getEmbeddings(const std::vector<std::string>& words) {
        std::vector<MatrixXld> emb_; 
        for (const auto& w_ : words) {
            emb_.push_back(getEmbedding(w_)); 
        } 
        return emb_;
    }

    size_t size() const {
        return data_ch2vec.size();
    }

private:
    int char_to_index(char ch) const {
        for (size_t i = 0; i < index2ch.size(); ++i)
            if (index2ch[i] == ch) return static_cast<int>(i);
        return -1;
    }
};


int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);
	Dictionary dic;

	dic.emb_size = 1;
	int i = 0;
	for(auto ch_ : "?<>/`јаЅб¬в√гƒд≈е®Є∆ж«з»и…й кЋлћмЌнќоѕп–р—с“т”у‘ф’х÷ц„чЎшўщЏъџы№ьЁэёюя€") {
		RowVectorXld vec_({{(double)i}});
		dic.push(ch_, vec_);
		i++;
	}

	/*Seq2SeqWithAttention_ForTrain test(1, 16, 16, 8, 1, dic.getEmbedding("/"), dic.getEmbedding("<`>"), 10, 4);

	std::vector<MatrixXld> input({
		dic.getEmbedding("ћд"), dic.getEmbedding("ћƒ"), dic.getEmbedding("мд"), dic.getEmbedding("мƒ"),
		dic.getEmbedding("ƒм"), dic.getEmbedding("ƒћ"), dic.getEmbedding("дм"), dic.getEmbedding("дћ")
		});
	std::vector<MatrixXld> output(1, dic.getEmbedding("ћ"));

    std::vector<std::vector<MatrixXld>> input_output({ {input[0], output[0]}, {input[1], output[0]}, 
        {input[2], output[0]}, {input[3], output[0]}, { input[4], output[0] }, 
        {input[5], output[0]}, { input[6], output[0] }, {input[7], output[0]} });

    test.UpdateAdamOptWithLogging(input_output, 1, 1000, 8, "test3", 1e-2); */
 
    Seq2SeqWithAttention_ForTrain test;
    test.Load("test3");
    
    std::vector<MatrixXld> input_t = dic.getEmbeddings({ "ћд", "ћƒ", "мд", "мƒ", "ƒм", "ƒћ", "дм", "дћ", "д?", "ƒ?"});
    std::vector<MatrixXld> output(1, dic.getEmbedding("ћ"));
    
    std::vector<std::vector<MatrixXld>> input_output({ {input_t[0], output[0]}, {input_t[1], output[0]},
        {input_t[2], output[0]}, {input_t[3], output[0]}, { input_t[4], output[0] },
        {input_t[5], output[0]}, { input_t[6], output[0] }, {input_t[7], output[0]}});
    
    test.UpdateAdamOptWithLogging(input_output, 1, 1000, "test3", 1e-2);
  
    Seq2SeqWithAttention_ForTrain test1;
    test1.Load("test3");
    
    std::string input;
    
    while(true){
        std::cin >> input;
    
        //std::cout << input;
    
        MatrixXld input_ = dic.getEmbedding(input);
    
        test1.Inference(std::vector<MatrixXld>(1, input_));
    
        MatrixXld output = test1.GetOutputs()[0].unaryExpr([](double x) { return std::round(x); });
    
        std::cout << dic.getWords(output) << std::endl;
    }
    
	return 0;
} 