DROP TABLE IF EXISTS user;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  age INTEGER NOT NULL,
  sex INTEGER NOT NULL
);
--- to do with database
-- age
-- 2. sex (1 = male; 0 = female)
-- 3. chest pain type (1=typical angin,2=atypical angina,3=non-anginal pain,4=asymptomatic)
-- 4. resting blood pressure
-- 5. serum cholestoral in mg/dl
-- 6. fasting blood sugar > 120 mg/dl
-- 7. resting electrocardiographic results (0=normal,1=having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV),2=showing probable or definite left
-- ventricular hypertrophy by Estes’ criteria)
-- 8. maximum heart rate achieved
-- 9. exercise induced angina
-- 10. oldpeak = ST depression induced by exercise relative to rest
-- 11. the slope of the peak exercise ST segment
-- 12. number of major vessels (0-3) colored by flourosopy
-- thal(Thalassemia): 3 = normal; 6 = fixed defect; 7 = reversable defect
-- 14. target: have disease or not (0=no,otherwise yes)
---