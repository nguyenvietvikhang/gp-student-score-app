import streamlit as st
import pandas as pd
import operator, math, random
from deap import base, creator, gp, tools, algorithms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="GP Dự đoán điểm học sinh", layout="centered")
st.markdown("""
<style>
    .main { background-color: #f7f7f7; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton button { background-color: #1f77b4; color: white; font-weight: bold; }
    .stSlider > div > div { color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

st.title("📘 Dự đoán điểm số học sinh bằng Lập trình Di truyền")

# --- Tải dữ liệu ---
uploaded_file = st.file_uploader("📂 Tải lên file CSV chứa dữ liệu học sinh (có cột 'final_score')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Xem trước dữ liệu:")
    st.dataframe(df.head())

    if 'final_score' not in df.columns:
        st.error("❌ File cần có cột 'final_score' làm nhãn đầu ra.")
    else:
        features = [col for col in df.columns if col != 'final_score']
        X = df[features].values.tolist()
        y = df['final_score'].values.tolist()

        # Thiết lập GP
        pset = gp.PrimitiveSet("MAIN", len(features))
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))
        for i, name in enumerate(features):
            pset.renameArguments(**{f"ARG{i}": name})

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def evalGP(ind):
            func = toolbox.compile(expr=ind)
            try:
                preds = [func(*row) for row in X]
                loss = sum((yt - yp) ** 2 for yt, yp in zip(y, preds)) / len(y)
                return (loss,)
            except:
                return (float("inf"),)

        toolbox.register("evaluate", evalGP)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=20))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=20))

        st.subheader("⚙️ Cấu hình mô hình")
        n_gen = st.slider("📈 Số thế hệ", 10, 100, 40)
        pop_size = st.slider("👥 Số cá thể", 10, 200, 100)

        if st.button("🚀 Chạy mô hình GP"):
            random.seed(42)
            pop = toolbox.population(n=pop_size)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("min", lambda x: min(v[0] for v in x))

            _, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, n_gen, stats=stats, halloffame=hof, verbose=False)

            best_expr = hof[0]
            st.success("✅ Biểu thức tốt nhất tìm được:")
            st.code(str(best_expr))

            func = toolbox.compile(expr=best_expr)
            try:
                preds = [func(*row) for row in X]
                mse = mean_squared_error(y, preds)
                mae = mean_absolute_error(y, preds)
                r2 = r2_score(y, preds)

                st.subheader("📐 Đánh giá mô hình")
                st.markdown(f"- **MSE**: `{mse:.4f}`")
                st.markdown(f"- **MAE**: `{mae:.4f}`")
                st.markdown(f"- **R² score**: `{r2:.4f}`")
            except:
                st.warning("⚠️ Không thể đánh giá mô hình do lỗi tính toán biểu thức.")

            st.subheader("📉 Biểu đồ tiến hóa (Loss qua từng thế hệ)")
            gens = range(len(log))
            if log and all('min' in entry for entry in log):
                losses = [entry['min'] for entry in log]
                fig, ax = plt.subplots()
                ax.plot(gens, losses, marker='o', color='#1f77b4')
                ax.set_xlabel("Thế hệ")
                ax.set_ylabel("Loss trung bình")
                ax.set_title("Tiến hóa biểu thức")
                st.pyplot(fig)
            else:
                st.warning("Không có dữ liệu để vẽ biểu đồ. Có thể do cá thể không hợp lệ.")

            st.info("🔍 Biểu thức này có thể dùng để dự đoán điểm số học sinh từ dữ liệu đầu vào.")
