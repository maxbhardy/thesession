#https://github.com/pgvector/pgvector?tab=readme-ov-file#docker
FROM pgvector/pgvector:pg17

COPY schema.sql /docker-entrypoint-initdb.d/

EXPOSE 5432