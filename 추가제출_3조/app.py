from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

#유저데이터 불러오기
u_data = pd.read_csv('user_data.csv')

#제품데이터 불러오기
p_data = pd.read_csv('product_data.csv')

# product_id를 기준으로 유저, 제품 데이터를 합쳐 df에 저장, user_id를 기준으로 오름차순 정렬
df = pd.merge(p_data, u_data, on='product_id').sort_values(by='user_id')
df.drop(['Unnamed: 5','Unnamed: 6'],axis=1, inplace=True)

# 각 유저가 제품에 남긴 평점들로 이루어진 데이터 프레임(df_users) 만들기 인덱스:user_id, 컬럼명:product_id
df_users = df.pivot_table('rating', index='user_id', columns='product_id')
df_users.fillna(0, inplace=True)

# 유저간 유사도 계산
cos_matrix = cosine_similarity(df_users.values,df_users.values)

# 위에서 계산한 cos_matrix를 데이터프레임으로 만듭니다.
df_users_cosine = pd.DataFrame(data=cos_matrix, index=df_users.index, columns=df_users.index)



def user_based_recommend(user_id, product_type):
    # 1. df_sers_cosine에서 입력한 아이디와 유사도 높은 5명을 sim_users로 설정
    sim_users = df_users_cosine.loc[user_id,:]
    sim_users[user_id] = 0
    sim_users = sim_users.sort_values(ascending = False).head(5)

    # 2. df에서 user_id가 sim_users의 인덱스와 일치하는 값들을 sim_user_df에 할당
    sim_user_df = df[df['user_id'].isin(sim_users.index)]

    # 3. sim_user_df에서 입력한 product_type과 일치하는 값들을 지정하고 rating을 기준으로 내림차순한 결과를 product에 할당
    products = sim_user_df[sim_user_df['product_type'] == product_type].sort_values(by = 'rating', ascending = False)

    # 4. products에서 rating 4점 이상인 값들만 다시 products에 할당
    products = products[products['rating'] >= 4]

    # 5. products에서 product_name이 중복인 것은 첫번째 값만 남김
    products.drop_duplicates(subset='product_name', keep='first', inplace=True)
    products.reset_index(drop=True)

    # 6. products의 product_name을 데이터 프레임 result로 만듦
    result = pd.DataFrame({f'나와 비슷한 사용자가 만족한 {product_type} 제품': products['product_name']})
    result.reset_index(drop=True, inplace=True)
    
    # 7. products의 product_name 컬럼을 데이터프레임 result로 만듦
    result = products['product_name'].tolist()

    title = f'나와 비슷한 사용자가 만족한 {product_type} 제품'

    return title, result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        product_type = request.form['product_type']
        title, result = user_based_recommend(user_id, product_type)
        return render_template('result.html', title=title, result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
