SELECT departmentid, AVG(salary)
FROM employee
GROUP BY DEPARTMENTID;


SELECT departmentid, name, salary, AGV(salary) OVER (PARTITION departmentid) as average_salary
FROM employee;
