#include "HeaderSeq2seqWithAttention.h"
#include <unordered_map>
#include <windows.h>
#ifdef max
#undef max
#endif 

namespace Dic {
    const std::string Kirr = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя";
    const std::string Lat = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz";
    const std::string Spec = R"( !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)";
    const std::string Num = "0123456789";
    static MatrixXld getEmbedding(const std::string& word_) {
        MatrixXld emb(word_.size(), 2); 
        for (size_t i = 0; i < word_.size(); i++) {
            auto& vec_first_col = emb(i, 0);
            auto& vec_second_col = emb(i, 1);
            if (Kirr.find(word_[i]) != std::string::npos) {
                vec_first_col = 1;
                vec_second_col = static_cast<double>(Kirr.find(word_[i]));
            }
            else if (Lat.find(word_[i]) != std::string::npos) {
                vec_first_col = 2;
                vec_second_col = static_cast<double>(Lat.find(word_[i]));
            }
            else if (Spec.find(word_[i]) != std::string::npos) {
                vec_first_col = 3;
                vec_second_col = static_cast<double>(Spec.find(word_[i]));
            }
            else if (Num.find(word_[i]) != std::string::npos) {
                vec_first_col = 4;
                vec_second_col = static_cast<double>(Num.find(word_[i]));
            }
            else {
                vec_first_col = 5;
                vec_second_col = 0;
            }
        }
        return emb;
    }
    static std::string getWords(const MatrixXld& mat) {
        std::string result;
        result.resize(mat.rows());
        //std::cout << mat << std::endl;
        for (Eigen::Index i = 0; i < mat.rows(); ++i) {
            if (mat(i, 0) < 0 || mat(i, 1) < 0) {
                std::abort();
            }
            size_t m0 = static_cast<size_t>(mat(i, 0));
            size_t m1 = static_cast<size_t>(mat(i, 1));
            bool m0_p = m0 != 1 && m0 != 2 && m0 != 3;
            bool m1_p = ((m0 == 1 && m1 >= Kirr.size()) || (m0 == 2 && m1 >= Lat.size()) || (m0 == 3 && m1 >= Spec.size()) || (m0 == 4 && m1 >= Num.size()));
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
                else if (m0 == 4) {
                    result[i] = Num[m1];
                }
            }
            else {
                result[i] = '$';
            }
        }
        return result;
    }
    static std::vector<MatrixXld> getEmbeddings(std::vector<std::string> words_) {
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
            throw std::invalid_argument("Размер вектора не совпадает с emb_size");

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
            throw std::out_of_range("Символ не найден в словаре");
        return data_ch2vec.at(ch_);
    }

    
    MatrixXld getEmbedding(const std::string& word_) const {
        MatrixXld emb(word_.size(), emb_size);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(word_.size()); ++i) {
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

    // Обратное отображение: вектор -> ближайший символ
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

    std::vector<MatrixXld> getEmbeddings(const std::vector<std::string>& words) const {
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
		dic.getEmbedding("Мд"), dic.getEmbedding("МД"), dic.getEmbedding("мд"), dic.getEmbedding("мД"),
		dic.getEmbedding("Дм"), dic.getEmbedding("ДМ"), dic.getEmbedding("дм"), dic.getEmbedding("дМ")
		});
	std::vector<MatrixXld> output(1, dic.getEmbedding("М"));

    std::vector<std::vector<MatrixXld>> input_output({ {input[0], output[0]}, {input[1], output[0]}, 
        {input[2], output[0]}, {input[3], output[0]}, { input[4], output[0] }, 
        {input[5], output[0]}, { input[6], output[0] }, {input[7], output[0]} });

    test.UpdateAdamOptWithLogging(input_output, 1, 1000, 8, "test3", 1e-2); */
 
    Seq2SeqWithAttention_ForTrain test;// (2, 16, 16, 8, 2, Dic::getEmbedding("!"), Dic::getEmbedding("</>"), 10, 50);
    test.Load("test_command_1");
    
    std::vector<MatrixXld> input_1 = Dic::getEmbeddings({"Сменить", "сменить", "СМенить", "сМенить", "Смен", "смен", "СМен", "сМен",
        "Change", "change", "CHange", "CHANGE", "СМЕНИТЬ"});
    std::vector<MatrixXld> output_1(1, Dic::getEmbedding("<COMMAND>CHANGE<COMMAND>"));
    
    std::vector<MatrixXld> input_2 = Dic::getEmbeddings({ "Да", "да", "ДА", "дА", "Yes", "yes", "YES", "YEs"});
    std::vector<MatrixXld> output_2(1, Dic::getEmbedding("<COMMAND>YES<COMMAND>"));

    std::vector<MatrixXld> input_3 = Dic::getEmbeddings({ "Нет", "нет", "НЕТ", "НЕт", "No", "no", "NO", "nO" });
    std::vector<MatrixXld> output_3(1, Dic::getEmbedding("<COMMAND>NO<COMMAND>"));

    
    std::vector<std::vector<MatrixXld>> input_output(input_1.size() + input_2.size() + input_3.size());

    for (size_t i = 0; i < input_1.size(); i++) {
        input_output[i] = { input_1[i], output_1[0]};
    }
    for (size_t i = 0; i < input_2.size(); i++) {
        input_output[input_1.size() + i] = { input_2[i], output_2[0] };
    }
    for (size_t i = 0; i < input_3.size(); i++) {
        input_output[input_1.size() + input_2.size() + i] = { input_3[i], output_3[0] };
    }

    
    test.UpdateAdamOptWithLogging(input_output, 2, 100, 8, "test_command_2", 1);
    
	return 0;
} 