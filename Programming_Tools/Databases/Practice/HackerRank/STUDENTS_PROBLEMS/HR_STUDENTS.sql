-- this is my solutions for the students family of problem on HACKER RANK

-- solution for this problem: https://www.hackerrank.com/challenges/the-report/problem?isFullScreen=true 

-- let's understand how to derive the grade from the marks firs 
DROP TABLE IF EXISTS TEMP ;
CREATE TABLE IF NOT EXISTS temp AS
(SELECT id, name, marks, (CASE WHEN marks = 100 THEN 10 ELSE FLOOR(marks / 10) + 1 END) as grade, 
(CASE WHEN  marks >= 70 THEN name ELSE NULL END) as ordered_name FROM students ORDER BY grade DESC,  ordered_name ASC , marks ASC);

SELECT (CASE WHEN grade >= 8 THEN name ELSE NULL END) AS name, grade, marks FROM temp;





