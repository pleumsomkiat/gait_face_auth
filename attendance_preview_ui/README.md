# Attendance Preview UI

UI จำลองสำหรับดูข้อมูลที่ระบบสแกนส่งเข้า API ก่อนหรือระหว่างบันทึกลงฐานข้อมูลจริง

## Run

```powershell
cd attendance_preview_ui
npm install
npm start
```

เปิดที่ `http://localhost:3000`

## Environment

- `MONGO_URI` ถ้าใส่ จะเปิดโหมดบันทึกลงฐานข้อมูลจริง
- ถ้าไม่ใส่ ระบบยังใช้หน้า preview ได้ และรองรับ `preview_only`
