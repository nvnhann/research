CREATE TABLE Employee (
	employee_id varchar(8) PRIMARY KEY,
	employee_name varchar(100),
	email varchar(100),
	hire_date DATE
);

INSERT INTO Employee(employee_id, employee_name, email, hire_date) SELECT DISTINCT emp_id, emp_nm, email, hire_dt FROM proj_stg;

CREATE TABLE Job (
	job_id SERIAL PRIMARY KEY,
	job_title varchar(100)
);

INSERT INTO Job(job_title) SELECT DISTINCT job_title FROM proj_stg;

CREATE TABLE Department (
    department_id SERIAL PRIMARY KEY,
	department_name varchar(100)
);
		
INSERT INTO Department(department_name) SELECT DISTINCT department_nm FROM proj_stg;

CREATE TABLE Salary (
	salary_id SERIAL PRIMARY KEY,
	salary INTEGER
);
			
INSERT INTO Salary(salary) SELECT salary FROM proj_stg;

CREATE TABLE Location (
	location_id SERIAL PRIMARY KEY,
	location varchar(100),
	state varchar(2),
	city varchar(50),
	address varchar(100)
);

INSERT INTO Location(location, state, city, address) SELECT DISTINCT location, state, city, address FROM proj_stg;

CREATE TABLE Education_level (
	education_id SERIAL PRIMARY KEY,
	education_level varchar(50)
);

INSERT INTO Education_level(education_level) SELECT DISTINCT education_lvl FROM proj_stg;

CREATE TABLE Employment (
	employee_id varchar(8), 
	location_id INTEGER, 
	department_id INTEGER, 
	salary_id INTEGER, 
	education_id INTEGER, 
	job_id INTEGER,
	manager_id varchar(8),
	started_date DATE, 
	ended_date DATE
);

CREATE VIEW manager AS 
SELECT 
  s.emp_id AS manager_id, 
  p.manager AS manager_name 
FROM 
  proj_stg AS p FULL 
  JOIN (
    SELECT 
      DISTINCT emp_id, 
      emp_nm 
    FROM 
      proj_stg 
    WHERE 
      emp_nm IN (
        SELECT 
          DISTINCT manager 
        FROM 
          proj_stg
      )
  ) AS s ON p.manager = s.emp_nm;


INSERT INTO Employment(
  employee_id, location_id, department_id, 
  salary_id, education_id, job_id, 
  manager_id, started_date, ended_date
)
SELECT 
  DISTINCT e.employee_id, 
  l.location_id, 
  d.department_id, 
  s.salary_id, 
  x.education_id, 
  j.job_id, 
  m.manager_id, 
  p.start_dt, 
  p.end_dt 
FROM 
  proj_stg AS p 
  JOIN employee AS e ON e.employee_id = p.emp_id 
  JOIN location AS l ON l.location = p.location 
  JOIN department AS d ON p.department_nm = d.department_name 
  JOIN salary AS s ON s.salary = p.salary 
  JOIN education_level AS x ON x.education_level = p.education_lvl 
  JOIN job AS j ON j.job_title = p.job_title 
  JOIN manager AS m ON p.manager = m.manager_name;

ALTER TABLE Employment ADD FOREIGN KEY (employee_id) REFERENCES Employee(employee_id);
ALTER TABLE Employment ADD FOREIGN KEY (location_id) REFERENCES location(location_id);
ALTER TABLE Employment ADD FOREIGN KEY (education_id) REFERENCES education_level(education_id);
ALTER TABLE Employment ADD FOREIGN KEY (job_id) REFERENCES job(job_id);
ALTER TABLE Employment ADD FOREIGN KEY (department_id) REFERENCES department(department_id);
ALTER TABLE Employment ADD FOREIGN KEY (salary_id) REFERENCES salary(salary_id);
ALTER TABLE Employment ADD FOREIGN KEY (manager_id) REFERENCES Employee(employee_id);

-- Question 1: Return a list of employees with Job Titles and Department Names

SELECT 
  e.employee_id, 
  j.job_title, 
  d.department_name 
FROM 
  employee e 
  JOIN employment f ON e.employee_id = f.employee_id 
  JOIN job j ON j.job_id = f.job_id 
  JOIN department d ON d.department_id = f.department_id;

-- Question 2: Insert Web Programmer as a new job title
INSERT INTO job(job_title) VALUES ('Web Programmer');

-- Question 3: Correct the job title from web programmer to web developer
UPDATE job SET job_title='Web Developer' WHERE job_title='Web Programmer';

-- Question 4: Delete the job title Web Developer from the database
DELETE FROM job WHERE job_title='Web Developer';

-- Question 5: How many employees are in each department?
SELECT 
  d.department_name, 
  COUNT(e.employee_id) 
FROM 
  department AS d 
  JOIN employment AS f ON d.department_id = f.department_id 
  JOIN employee AS e ON e.employee_id = f.employee_id 
GROUP BY 
  d.department_name;

-- Question 6: Write a query that returns current and past jobs (include employee name, job title, department, manager name, start and end date for position) for employee Toni Lembeck.

WITH manager_info AS (
  SELECT 
    DISTINCT e.employee_id as manager_id, 
    e.employee_name AS manager 
  FROM 
    employee e 
    JOIN employment em ON e.employee_id = em.manager_id
) 
SELECT 
  DISTINCT e.employee_name, 
  j.job_title, 
  d.department_name, 
  m.manager, 
  f.started_date, 
  f.ended_date 
FROM 
  employee e 
  JOIN employment f ON e.employee_id = f.employee_id 
  JOIN department d ON d.department_id = f.department_id 
  JOIN manager_info m ON m.manager_id = f.manager_id 
  JOIN job j ON j.job_id = f.job_id 
WHERE 
  e.employee_name = 'Toni Lembeck';

-- Question 7: Describe how you would apply table security to restrict access to employee salaries using an SQL server.
