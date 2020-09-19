BASELINE DỰ ĐOÁN GIÁ CHO BẤT ĐỘNG SẢN 
SỬ DỤNG MODEL LINEAR REGRESSION 

# MÔ TẢ : 
Ứng dụng mô hình Linear Regression của Machine Learning nhằm dự đoán giá cho các bất động sản thuộc một khu vực nhất định trong một khoảng thời gian được định trước (giá trị đầu vào) 


# CÁC BƯỚC : 
	1. Trích xuất thông tin của các bài post 
	2. Lọc ra các bài post trong vùng cần dự đoán giá 
	3. Tiền xử lí data trước khi xây dựng mô hình dự đoán giá : 
		- Xử lí các bài post bị null area ( gọi api make_up_alternative ) 
		- Xử lí các bài post bị mull giá bán 
		- Tính giá bán trên m2 thay vì full price
		- Xử lí các bài post null post_date và các bài post trùng post_date 
	4. Bắt đầu xây dựng mô hình dự đoán giá : 
		- X : post_date 
		- y: price_per_m2 	
		- Sử dụng ShuffleSplit thuộc sklearn.model_selection tách thành train set và test set 
		- Dùng train set để xây dựng model 
		- Dùng test set để đánh giá độ chính xác của model 
		- Áp dụng model dùng dự đoán giá trong khoảng thời gian trong tương lai 

# ĐÁNH GIÁ : 
Trước mắt chưa test độ chính xác, nhưng đánh giấ ban đầu sẽ không cao do đầu vào chưa chi tiết và ít có sự liên quan tói giá 
