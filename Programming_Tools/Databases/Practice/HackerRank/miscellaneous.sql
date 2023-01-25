-- this is my solution to the problem: https://www.hackerrank.com/challenges/the-blunder/problem?isFullScreen=true
-- the following links might be needed: 1. https://www.w3schools.com/sql/func_mysql_cast.asp 
--										2. 	

SELECT c.company_code, c.founder, l.lead_count, s.senior_count, 
    m.manager_count, e.e_count
FROM COMPANY as c 
LEFT JOIN (SELECT company_code, COUNT(lead_manager_code) as lead_count FROM LEAD_MANAGER GROUP BY company_code) as l
ON c.company_code = l.company_code
GROUP BY c.company_code;

