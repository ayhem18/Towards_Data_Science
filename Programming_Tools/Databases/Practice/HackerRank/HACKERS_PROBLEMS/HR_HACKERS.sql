DROP TABLE IF EXISTS challenges;
DROP TABLE IF EXISTS hackers;
CREATE TABLE IF NOT EXISTS hackers (
hacker_id INT NOT NULL AUTO_INCREMENT, 
name varchar(10), 
primary KEY(hacker_id)
);

INSERT INTO hackers (name) VALUES
	("ax"),
    ("ay"),
    ("cd"),
    ("de"),
    ("ea"),
    ("fa"),
    ("af"), 
    ('axy'),
    ('azb'),
    ('zz'),
    ('yed'),
    ('zy');
    
SELECT * FROM hackers;    
    
DROP TABLE IF EXISTS challenges;
CREATE TABLE IF NOT EXISTS challenges (
challenge_id int AUTO_INCREMENT, 
hacker_id int,

-- define the primary key as the combination of the two ids
PRIMARY KEY (challenge_id),

-- add the foreign key constraint
FOREIGN KEY(hacker_id)
REFERENCES hackers (hacker_id) ON DELETE CASCADE
);

INSERT INTO challenges (hacker_id) VALUES
(10), (10), (10), (10), (10), (10), (10), (10), 
(1),(1),(1),(1),(1),(1),(1),(1),  
(9),(9),(9),(9),(9),(9),(9),(9), 
(2),(2),(2),(2),(2),(2),(2),
(3),(3),(3),(3),(3),(3),(3),
(4),(4),(4),(4),(4),(4),(4),
(5),(5),(5),(5),(5),(5),(5),
(6),
(7), (7),
(8), (8), (8);

SELECT * FROM challenges;