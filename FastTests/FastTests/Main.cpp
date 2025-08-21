#include <iostream>

template<typename T>
class SharedPtr;

template<typename T>
class WeakPtr;


template<typename T>
class SharedPtr{
private:
	struct CountsForPtr {
		size_t smallptr_counts = 0;
		size_t bigptr_counts = 0;
	};
	CountsForPtr* counts_ = nullptr;
	T* ptr_ = nullptr;
public:
	SharedPtr(T* ptr__){
		if (ptr__ == nullptr) {
			throw std::invalid_argument("Pointer is null");
		}
		this->ptr_ = ptr__;
		this->counts_ = new CountsForPtr;
		this->counts_->bigptr_counts = 1;
	}
	SharedPtr(SharedPtr<T> && ptr__) {
		if (ptr__ == nullptr) {
			throw std::invalid_argument("Pointer is null");
		}
		this->ptr_ = ptr__->ptr_;
		this->counts_ = ptr__->counts_;
		this->counts_->bigptr_counts++;
	}
	SharedPtr(WeakPtr<T>&& ptr_) {
		if (ptr_.ptr_ == nullptr) {
			throw std::invalid_argument("Pointer is null");
		}
		this->ptr_ = ptr_.ptr_;
		this->counts_ = ptr_.counts;
		this->counts_->smallptr_counts++;
	}
	void operator=(T* ptr__) {
		*this = SharedPtr::SharedPtr(ptr__);
	}
	void operator=(SharedPtr<T>&& ptr_) {
		*this = SharedPtr::SharedPtr(ptr_);
	}
	void operator=(WeakPtr<T>&& ptr_) {
		*this = SharedPtr::SharedPtr(ptr_);
	}
	~SharedPtr() {
		if(this->counts_->bigptr_counts > 0){
			this->counts_->bigptr_counts--;		
		}
		if (this->counts_->bigptr_counts == 0) {
			delete this->ptr_;
			if(this->counts_->smallptr_counts == 0){
				delete this->counts_;
				this->counts_ = nullptr;
			}
		}
		this->ptr_ = nullptr;
	}
	T& operator* () {
		return *this->ptr_;
	}
	bool operator==(const SharedPtr<T> & ptr_) {
		if (ptr_ == this->ptr_) {
			return true;
		}
		else {
			return false;
		}
	}
	T* get() {
		return this->ptr_;
	}
	void clean() {
		this->ptr_ = nullptr;
		if (this->counts_->bigptr_counts > 0) {
			this->counts_->bigptr_counts--;
		}
	}
	void reset(const SharedPtr<T> && ptr_) {
		this->clean();
		*this = ptr_;
	}
};


template<typename T>
class WeakPtr {
	//friend class SharedPtr<T>;
private:
	struct CountsForPtr {
		size_t smallptr_counts = 0;
		size_t bigptr_counts = 0;
	};
	CountsForPtr* counts_ = nullptr;
	T* ptr_ = nullptr;
public:
	bool contaned() {
		if (this->ptr_ == nullptr) {
			return false;
		}
		else {
			return true;
		}
	}
	WeakPtr(WeakPtr<T>&& ptr__) {
		if (ptr__ == nullptr) {
			throw std::invalid_argument("Pointer is null");
		}
		this->ptr_ = ptr__->ptr_;
		this->counts_ = ptr__->counts_;
		this->counts_->smallptr_counts++;
	}
	WeakPtr(SharedPtr<T>&& ptr_) {
		if (ptr_.ptr_ == nullptr) {
			throw std::invalid_argument("Pointer is null");
		}
		this->ptr_ = ptr_.ptr_;
		this->counts_ = ptr_.counts_;
		this->counts_->smallptr_counts++;
	}
	void operator=(SharedPtr<T>&& ptr__) {
		*this = WeakPtr::WeakPtr(ptr__);
	}
	void operator=(WeakPtr<T>&& ptr_) {
		*this = WeakPtr::WeakPtr(ptr_);
	}
	~WeakPtr() {
		if (this->counts_->smallptr_counts > 0) {
			this->counts_->smallptr_counts--;
		}
		if (this->counts_->smallptr_counts == 0 && this->counts_->bigptr_counts == 0) {
			delete this->counts_;
			this->counts_ == nullptr;
		}
	}
	bool operator==(const WeakPtr<T>& ptr_) {
		if (ptr_ == this->ptr_) {
			return true;
		}
		else {
			return false;
		}
	}
	void reset(WeakPtr<T>&& ptr_) {
		this->clean();
		*this = ptr_;
	}
	void reset(SharedPtr<T>&& ptr_) {
		this->clean();
		*this = ptr_;
	}
};

int main() {
	SharedPtr<int> ptr = new int(10);
	WeakPtr<int> wptr(ptr);
	std::cout << *ptr << std::endl;
	return 0;
}