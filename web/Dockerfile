FROM node:hydrogen-bullseye AS builder

WORKDIR /frontend

RUN npm config set strict-ssl false
RUN npm config set registry http://registry.npmjs.org/

COPY frontend/ /frontend/

RUN npm install

RUN npm run build

#########=========> 

FROM node:hydrogen-bullseye AS server

WORKDIR /app

# We only require these 5 folders/files for nextjs apps in production
COPY --from=builder /frontend/next.config.js ./
COPY --from=builder /frontend/public ./public
COPY --from=builder /frontend/build ./build
COPY --from=builder /frontend/node_modules ./node_modules
COPY --from=builder /frontend/package.json ./package.json

CMD [ "npm", "run", "start" ]