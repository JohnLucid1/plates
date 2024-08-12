FROM python:3.12.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=another.py

# Run the application
# CMD ["python", "model.py"]
# CMD [ "executable" ]
CMD ["flask", "run", "--host=0.0.0.0"]
