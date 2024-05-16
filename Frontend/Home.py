import streamlit as st

st.set_page_config(page_title="BinGenius", layout="wide")

st.title("Home")

st.markdown("""<p style='font-size:25px;'> In today's fast-paced world,  sustainable waste management is no longer an option, it's a 
            necessity. Traditional methods often face challenges due to public confusion about what can be recycled.  This is where 
            innovation steps in! </p>""", unsafe_allow_html=True)

st.markdown("""<p style='font-size:25px;'> This project tackles recycling with a user-friendly approach. Our multimodal machine 
            learning system analyzes video input to  predict whether an object is recyclable. Simply upload a video of an item, 
            and our system will guide you towards the right recycling bin. </p>""", unsafe_allow_html=True)

st.header("About Us")

st.markdown("""<p style='font-size:25px;'> We are a passionate group of four individuals driven by a shared interest in the exciting 
            world of data science, artificial intelligence (AI), and machine learning (ML). We believe these fields hold immense potential 
            to tackle real-world challenges and create a positive impact. </p>""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.markdown("<p style='font-size:20px;'><b> Ananya Sachan </b></p>", unsafe_allow_html=True)
col1.write("Contribution: Data Cleaning, EDA and Presentation")
col1.markdown("Connect with Ananya on [LinkedIn](www.linkedin.com/in/ananya-sachan-7ba84924a)", unsafe_allow_html=True)

col2.markdown("<p style='font-size:20px;'><b> Divya Shanbhag </b></p>", unsafe_allow_html=True)
col2.write("Contribution: Data Collection, Image (Preprocessing), Model Building, Deployment and Testing")
col2.markdown("Connect with Divya on [LinkedIn](linkedin.com/in/divya-shanbhag-b86bb9264)", unsafe_allow_html=True)

col3.markdown("<p style='font-size:20px;'><b> Harshita Vyas </b></p>", unsafe_allow_html=True)
col3.write("Contribution: Data Cleaning, Audio (Preprocessing and Feature Extraction), Model Building and Presentation")
col3.markdown("Connect with Harshita on [LinkedIn](https://www.linkedin.com/in/harshitavyas04/)", unsafe_allow_html=True)

col4.markdown("<p style='font-size:20px;'><b> Jaanvi Das </b></p>", unsafe_allow_html=True)
col4.write("Contribution: Data Cleaning, Image (Feature Extraction), Model Building and Poster")
col4.markdown("Connect with Jaanvi on [LinkedIn](https://www.linkedin.com/in/jaanvidas/)", unsafe_allow_html=True)

