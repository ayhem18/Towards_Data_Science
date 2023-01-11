DROP TABLE IF EXISTS station ;
CREATE TABLE IF NOT EXISTS station (
id INT AUTO_INCREMENT,
city VARCHAR(21), 
state VARCHAR(2),
lat_n INT,
long_w INT,
PRIMARY KEY (id)
);

INSERT INTO station (city, state, lat_n, long_w) VALUES 
	('a', 'A', 11, 8), 
	('b', 'A', 23, 7), 
	('c', 'A', 3, 13), 
	('d', 'A', 20, 1),
	('e', 'A', 5, 24),
    ('e', 'B', 500, 24);
    


DROP TABLE IF EXISTS students ;

CREATE TABLE IF NOT EXISTS students (
id INT AUTO_INCREMENT,
name varchar(10), 
marks INTEGER,
PRIMARY KEY(id) 
);

INSERT INTO students (name, marks) VALUES
	('ab', 24),
	('ac', 0),
	('aa', 55),
    ('ar', 51),
	('ax', 54),
	('xa', 80),
    ('ar', 81),
	('ax', 94),
	('xa', 91),    
	('yz', 100);

    
    
    
    

 

    