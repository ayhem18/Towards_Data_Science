DROP TABLE IF EXISTS salary_1;
DROP TABLE IF EXISTS salary_2;
DROP TABLE IF EXISTS salary_3;

CREATE TABLE IF NOT EXISTS salary_1 AS (SELECT d.id as d_id, d.name as d_name, MAX(salary) as salary_1 
FROM department as d, employee as e
WHERE d.id = e.departmentid 
GROUP BY d.id, d.name);

-- SELECT * FROM salary_1;


CREATE TABLE IF NOT EXISTS salary_2 AS (
    SELECT d_id, d_name, MAX(salary) as salary_2
    FROM salary_1 
    LEFT JOIN employee as e -- left join as certain department might have only one unique value (they should still preserved in this case)
    ON salary_1.d_id = e.departmentid AND e.salary < salary_1.salary_1
    GROUP BY d_id, d_name
);

-- SELECT * FROM salary_2;

-- extending the same idea of salary_3 table to 
CREATE TABLE IF NOT EXISTS salary_3 AS (
    SELECT d_id, d_name, MAX(salary) as salary_3
    FROM salary_2 
    LEFT JOIN employee as e
    ON salary_2.d_id = e.departmentid AND e.salary < salary_2.salary_2
    GROUP BY d_id, d_name
);


-- SELECT * FROM salary_3;


SELECT d_name as "Department", e.name as "Employee", e.salary as "Salary"
FROM employee as e, salary_3 
WHERE d_id  = e.departmentid and (e.salary >= salary_3.salary_3 OR salary_3.salary_3 IS NULL);


-- better solution
COU





