FROM stablebaselines/stable-baselines3-cpu:latest
COPY . .
RUN pip install --upgrade pip==21.0.1 \
    && pip install --no-cache-dir -e . \
    && pip install tensorboardX
EXPOSE 6006
ENTRYPOINT ["tail", "-f", "/dev/null"]