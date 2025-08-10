#include "HeaderSeq2seqWithAttention.h"
#include <unordered_map>
#include <windows.h>
#ifdef max
#undef max
#endif 

namespace Dic {
    std::string Kirr = "јаЅб¬в√гƒд≈е®Є∆ж«з»и…й кЋлћмЌнќоѕп–р—с“т”у‘ф’х÷ц„чЎшўщЏъџы№ьЁэёюя€";
    std::string Lat = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";
    std::string Spec = "!?.,;:@#%^~`*/-+=<>[](){}'"/* + '"'*/;
    MatrixXld getEmbedding(const std::string& word_) {
        MatrixXld emb(word_.size(), 2);
        for (size_t i = 0; i < word_.size(); i++) {
            if (Kirr.find(word_[i]) != std::string::npos) {
                emb(i, 0) = 1;
                emb(i, 1) = Kirr.find(word_[i]);
            }
            else if (Lat.find(word_[i]) != std::string::npos) {
                emb(i, 0) = 2;
                emb(i, 1) = Lat.find(word_[i]);
            }
            else if (Spec.find(word_[i]) != std::string::npos) {
                emb(i, 0) = 3;
                emb(i, 1) = Spec.find(word_[i]);
            }
            else {
                emb(i, 0) = 4;
                emb(i, 1) = 0;
            }
        }
        return emb;
    }
    std::string getWords(const MatrixXld& mat) {
        std::string result;
        result.resize(mat.rows());
        //std::cout << mat << std::endl;
        for (Eigen::Index i = 0; i < mat.rows(); ++i) {
            if (mat(i, 0) < 0 || mat(i, 1) < 0) {
                std::abort();
            }
            size_t m0 = mat(i, 0);
            size_t m1 = mat(i, 1);
            bool m0_p = m0 != 1 && m0 != 2 && m0 != 3;
            bool m1_p = (m0 == 1 && m1 >= Kirr.size()) || (m0 == 2 && m1 >= Lat.size()) || (m0 == 3 && m1 >= Spec.size());
            if(!m0_p && !m1_p){
                if (m0 == 1) {
                    result[i] = Kirr[m1];
                }
                else if (m0 == 2) {
                    result[i] = Lat[m1];
                }
                else if (m0 == 3) {
                    result[i] = Spec[m1];
                }
            }
            else {
                result[i] = '$';
            }
        }
        return result;
    }
    std::vector<MatrixXld> getEmbeddings(std::vector<std::string> words_) {
        std::vector<MatrixXld> result;
        for (const auto & w_ : words_ ) {
            result.push_back(getEmbedding(w_));
        }
        return result;
    }
}
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
 
    //Seq2SeqWithAttention_ForTrain test(2, 16, 16, 8, 2, Dic::getEmbedding("!"), Dic::getEmbedding("</>"), 10, 10);
    ////test.Load("test6");
    //
    //std::vector<MatrixXld> input_t = Dic::getEmbeddings({ "ћд", "ћƒ", "мд", "мƒ", "ƒм", "ƒћ", "дм", "дћ", "д?", "ƒ?"});
    //std::vector<MatrixXld> output(1, Dic::getEmbedding("ћ"));
    //
    //std::vector<std::vector<MatrixXld>> input_output({ {input_t[0], output[0]}, {input_t[1], output[0]},
    //    {input_t[2], output[0]}, {input_t[3], output[0]}, { input_t[4], output[0] },
    //    {input_t[5], output[0]}, { input_t[6], output[0] }, {input_t[7], output[0]}, {input_t[8], output[0]}, {input_t[9], output[0]} });
    //
    //test.UpdateAdamOptWithLogging(input_output, 10, 500, "test7", 1e-2);

    Seq2SeqWithAttention_ForTrain test1;
    test1.Load("test7");
    
    std::string input;
    
    while(true){
        std::cin >> input;
    
        //std::cout << Dic::getEmbedding("</>");
    
        MatrixXld input_ = Dic::getEmbedding(input);
    
        test1.Inference(std::vector<MatrixXld>(1, input_));
    
        //std::cout << test1.GetOutputs()[0];
    
        MatrixXld output = test1.GetOutputs()[0].unaryExpr([](double x) { return std::round(x); });
    
        std::cout << Dic::getWords(output) << std::endl;
    }
    
	return 0;
} 