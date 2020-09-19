import psycopg2
import pandas as pd


# def get_price_construction():
#     price_construct_list = []

#     #########################################
#     # Kết nối lên server lấy các thông tin cần thiết
#     #########################################

#     try:
#         connection = psycopg2.connect(
#             user='adminsharkland',
#             password='atomic181',
#             host='localhost',
#             port=5432,
#             database='sharklanddb',
#         )

#         cursor = connection.cursor()

#         cursor = connection.cursor()
#         cursor.execute('select price from public.price_construction')
#         p_construct = cursor.fetchall()
#         price_construct_list = [p[0] for p in p_construct]

#     except psycopg2.Error as e:
#         print("Cannot conncet to database")

#     return price_construct_list


'''
#####################################
# CÔNG THỨC TÁCH GIÁ ĐẤT TỪ GIÁ BÁN 
#####################################

Case 1: Nếu realestate_type = 1 hoặc = 9 hoặc = 10 có nghĩa là(đất| đất nông nghiệp|đất nến)
Giá đất /m2 = Giá bán / diện tích đất

Case 2: Nếu realestate_type = 5 hoặc  =  7 (Nhà xưởng hoặc dãy phòng trọ)
Giá đất  = Giá bán - (Giá xây  thô đơn giản * Số lầu * diện tích xây)
Giá đất/m2 = Giá đất / diện tích đất

Case 3: Nếu realestate_type = 3 (Chung cư/căn hộ)
Giá trên m2 = Giá bán / diện tích

Case 4: Nếu realestate_type = 2 (Nhà riêng)

Case 4.1: Nếu floor <= 2
Giá đất  = Giá bán - (Giá xây thô đơn giản * Số lầu  * diện tích xây)
Case 4.2: Nếu floor > 2 và floor < 5
Giá đất  = Giá bán - (Giá xây vừa đủ tiện ích * Số lầu * diện tích xây)
Case 4.3: Nếu floor >= 5
Giá đất  = Giá bán - (Giá xây dựng sang trọng * Số lầu * diện tích xây)

Giá đất/m2 = Giá đất / diện tích đất

Case 5: Các trường hợp còn lại cho bằng 0

'''


def convert_price_sell(realestate_type, price_sell, area_cal, floor) -> float:
    price_construction = [3500000, 4500000, 5500000]
    # realestate_type = post['realestate-type']
    # price_sell = post['price_sell']
    # area_cal = post['area_cal']
    # floor = post['floor']
    if (realestate_type == 1) | (realestate_type == 9) | (realestate_type == 10):
        price_per_m2 = round(price_sell/area_cal, 2)
    elif (realestate_type == 5) | (realestate_type == 7):
        land_price = price_sell - (price_construction[0] * floor * area_cal)
        price_per_m2 = round(land_price / area_cal)
    elif (realestate_type == 3):
        price_per_m2 = round(price_sell / area_cal, 2)
    elif (realestate_type == 2):
        if floor <= 2:
            land_price = price_sell - \
                (price_construction[0] * floor * area_cal)
        elif (floor >= 2) and (floor <= 5):
            land_price = price_sell - \
                (price_construction[1] * floor * area_cal)
        elif floor >= 5:
            land_price = price_sell - \
                (price_construction[2] * floor * area_cal)
        price_per_m2 = round(land_price / area_cal, 2)
    else:
        price_per_m2 = 0
    return price_per_m2


def get_days_number(start_date: int, year: int, period: str):
    
    month = start_date.month
    odd = [1,3,5,7,8,10,12]
    even = [4,6,9,11]
    days = 0

    for  month_th in range(1, len(period)):
        if month in odd:
            days += 31
        if month in even:
            days += 30
        if month == 'thang2':
            if check_leap_year(year) == True:
                days += 29
            else:
                days += 28
    return days


def check_leap_year(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


# try:
#     connection = psycopg2.connect(host='localhost', dbname='sharklanddb',
#                                   user='adminsharkland', port=7001, password='atomic181')
#     posts_info = pd.read_sql("select * from post", connection)
#     posts_info


# except (psycopg2.Error) as e:
#     print(e)
# finally:

#     connection.close()
