DROP TABLE IF EXISTS employee ;
DROP TABLE IF EXISTS department;

CREATE TABLE IF NOT EXISTS department (
id INT NOT NULL AUTO_INCREMENT, 
name VARCHAR(10),
PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS employee (
id INT not null AUTO_INCREMENT,
name VARCHAR(10), 
SALARY INT,
DEPARTMENTID INT,
PRIMARY KEY(id),
FOREIGN KEY (departmentid) REFERENCES department (id) ON DELETE CASCADE
);

-- 3 departments would be good enough
INSERT INTO department (name) VALUES ('sales'), ('IT'), ('legal');

-- let's add some employees

INSERT INTO employee (NAME, SALARY, DEPARTMENTID) VALUES 
		('a', 40, 1),
		('b', 45, 1),
		('c', 50, 1),
		('d', 50, 1),
		('da', 60, 1),
		('ds', 60, 1),
		('dc', 85, 1),
		('de', 90, 1),

		('x', 60, 2),
		('x1', 65, 2),
		('x3', 65, 2),
		('x4', 65, 2),
		('x2', 70, 2),
		('ay', 80, 2),
		('boss', 80, 2),
		('s1', 70, 3),
		('s2', 70, 3),
		('s3', 75, 3),
		('s4', 75, 3),
		('s5', 75, 3);

        
        
        
        
        
        
        