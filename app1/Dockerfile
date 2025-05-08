# Dockerfile
FROM openjdk:11-jdk-slim

# # -- 1) System deps & Python 3.10 --
# RUN apt-get update && \
#     apt-get install -y python3 python3-pip wget curl && \
#     ln -sf /usr/bin/python3 /usr/bin/python

# 1) Install Python3 & pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# -- 2) Install pyspark, streamlit and plotting libs --
RUN pip3 install --no-cache-dir \
    pyspark==3.5.5 \
    streamlit \
    matplotlib \
    numpy \
    pandas

# -- 4) Copy your app into the container --
WORKDIR /app
COPY webui.py .
COPY model_stage4_merged_Trans_way_new ./model_stage4_merged_Trans_way_new
COPY model_stage2_Trans_way ./model_stage2_Trans_way

# -- 5) Expose Streamlit port and set entrypoint --
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "webui.py", "--server.headless", "true"]
