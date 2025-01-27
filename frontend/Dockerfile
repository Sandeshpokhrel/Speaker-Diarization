# Step 1: Use Node.js for building the React app
FROM node:16 as build

# Set the working directory inside the container
WORKDIR /app

# Copy the package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the app and build the React app
COPY . ./
RUN npm run build

# Step 2: Use NGINX to serve the built files
FROM nginx:latest

# Copy the React build output to the NGINX directory
COPY --from=build /app/build /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start NGINX
CMD ["nginx", "-g", "daemon off;"]