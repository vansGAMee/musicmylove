FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src/lib ./src/lib
COPY services ./services
EXPOSE 8787
CMD ["npm", "run", "gateway"]
