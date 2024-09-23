# Use the VIPAS AI Streamlit base image
FROM vipasai/vps-streamlit-base:1.0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]  # Change 'app.py' if your main file is named differently
