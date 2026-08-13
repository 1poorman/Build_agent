
# PostgreSQL
## 查看所有记录
PGPASSWORD=postgres psql -h localhost -U postgres -d langchain_memory \
  -c "SELECT id, session_id, role, LEFT(content, 100) FROM chat_messages ORDER BY id;"

## 查看完整内容（不截断）
PGPASSWORD=postgres psql -h localhost -U postgres -d langchain_memory \
  -c "SELECT * FROM chat_messages ORDER BY id;"

## 统计
PGPASSWORD=postgres psql -h localhost -U postgres -d langchain_memory \
  -c "SELECT role, COUNT(*) FROM chat_messages GROUP BY role;"

## 交互式模式
PGPASSWORD=postgres psql -h localhost -U postgres -d langchain_memory
