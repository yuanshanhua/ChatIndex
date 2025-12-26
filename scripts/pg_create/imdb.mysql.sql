DROP DATABASE IF EXISTS imdb;
CREATE DATABASE imdb;
USE imdb;

DROP TABLE IF EXISTS aka_name;
CREATE TABLE aka_name (
    id INT NOT NULL,
    person_id INT NOT NULL,
    name TEXT,
    imdb_index VARCHAR(3),
    name_pcode_cf VARCHAR(11),
    name_pcode_nf VARCHAR(11),
    surname_pcode VARCHAR(11),
    md5sum VARCHAR(65),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS aka_title;
CREATE TABLE aka_title (
    id INT NOT NULL,
    movie_id INT NOT NULL,
    title TEXT,
    imdb_index VARCHAR(4),
    kind_id INT NOT NULL,
    production_year INT,
    phonetic_code VARCHAR(5),
    episode_of_id INT,
    season_nr INT,
    episode_nr INT,
    note VARCHAR(72),
    md5sum VARCHAR(32),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS cast_info;
CREATE TABLE cast_info (
    id INT NOT NULL,
    person_id INT NOT NULL,
    movie_id INT NOT NULL,
    person_role_id INT,
    note TEXT,
    nr_order INT,
    role_id INT NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS char_name;
CREATE TABLE char_name (
    id INT NOT NULL,
    name TEXT,
    imdb_index VARCHAR(2),
    imdb_id INT,
    name_pcode_nf VARCHAR(5),
    surname_pcode VARCHAR(5),
    md5sum VARCHAR(32),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS comp_cast_type;
CREATE TABLE comp_cast_type (
    id INT NOT NULL,
    kind VARCHAR(32) NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS company_name;
CREATE TABLE company_name (
    id INT NOT NULL,
    name TEXT,
    country_code VARCHAR(6),
    imdb_id INT,
    name_pcode_nf VARCHAR(5),
    name_pcode_sf VARCHAR(5),
    md5sum VARCHAR(32),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS company_type;
CREATE TABLE company_type (
    id INT NOT NULL,
    kind VARCHAR(32),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS complete_cast;
CREATE TABLE complete_cast (
    id INT NOT NULL,
    movie_id INT,
    subject_id INT NOT NULL,
    status_id INT NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS info_type;
CREATE TABLE info_type (
    id INT NOT NULL,
    info VARCHAR(32) NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS keyword;
CREATE TABLE keyword (
    id INT NOT NULL,
    keyword TEXT NOT NULL,
    phonetic_code VARCHAR(5),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS kind_type;
CREATE TABLE kind_type (
    id INT NOT NULL,
    kind VARCHAR(15),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS link_type;
CREATE TABLE link_type (
    id INT NOT NULL,
    link VARCHAR(32) NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS movie_companies;
CREATE TABLE movie_companies (
    id INT NOT NULL,
    movie_id INT NOT NULL,
    company_id INT NOT NULL,
    company_type_id INT NOT NULL,
    note TEXT,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS movie_info_idx;
CREATE TABLE movie_info_idx (
    id INT NOT NULL,
    movie_id INT NOT NULL,
    info_type_id INT NOT NULL,
    info TEXT NOT NULL,
    note VARCHAR(1),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS movie_keyword;
CREATE TABLE movie_keyword (
    id INT NOT NULL,
    movie_id INT NOT NULL,
    keyword_id INT NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS movie_link;
CREATE TABLE movie_link (
    id INT NOT NULL,
    movie_id INT NOT NULL,
    linked_movie_id INT NOT NULL,
    link_type_id INT NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS name;
CREATE TABLE name (
    id INT NOT NULL,
    name TEXT NOT NULL,
    imdb_index VARCHAR(9),
    imdb_id INT,
    gender VARCHAR(1),
    name_pcode_cf VARCHAR(5),
    name_pcode_nf VARCHAR(5),
    surname_pcode VARCHAR(5),
    md5sum VARCHAR(32),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS role_type;
CREATE TABLE role_type (
    id INT NOT NULL,
    role VARCHAR(32) NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS title;
CREATE TABLE title (
    id INT NOT NULL,
    title TEXT,
    imdb_index VARCHAR(5),
    kind_id INT NOT NULL,
    production_year INT,
    imdb_id INT,
    phonetic_code VARCHAR(5),
    episode_of_id INT,
    season_nr INT,
    episode_nr INT,
    series_years VARCHAR(49),
    md5sum VARCHAR(32),
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS movie_info;
CREATE TABLE movie_info (
    id INT NOT NULL,
    movie_id INT NOT NULL,
    info_type_id INT NOT NULL,
    info TEXT,
    note TEXT,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

DROP TABLE IF EXISTS person_info;
CREATE TABLE person_info (
    id INT NOT NULL,
    person_id INT NOT NULL,
    info_type_id INT NOT NULL,
    info TEXT,
    note TEXT,
    PRIMARY KEY (id)
) ENGINE=InnoDB;

CREATE INDEX idx_movie_keyword_movie_id ON movie_keyword USING BTREE (movie_id);
CREATE INDEX idx_movie_companies_company_id ON movie_companies USING BTREE (company_id);
CREATE INDEX idx_movie_companies_company_type_id ON movie_companies USING BTREE (company_type_id);
CREATE INDEX idx_movie_info_idx_movie_id ON movie_info_idx USING BTREE (movie_id);
CREATE INDEX idx_aka_name_person_id ON aka_name USING BTREE (person_id);
CREATE INDEX idx_movie_info_idx_info_type_id ON movie_info_idx USING BTREE (info_type_id);
CREATE INDEX idx_person_info_person_id ON person_info USING BTREE (person_id);
CREATE INDEX idx_title_kind_id ON title USING BTREE (kind_id);
CREATE INDEX idx_cast_info_movie_id ON cast_info USING BTREE (movie_id);
CREATE INDEX idx_movie_companies_movie_id ON movie_companies USING BTREE (movie_id);
CREATE INDEX idx_cast_info_person_id ON cast_info USING BTREE (person_id);
CREATE INDEX idx_cast_info_person_role_id ON cast_info USING BTREE (person_role_id);
CREATE INDEX idx_movie_keyword_keyword_id ON movie_keyword USING BTREE (keyword_id);
CREATE INDEX idx_movie_info_movie_id ON movie_info USING BTREE (movie_id);
CREATE INDEX idx_aka_title_movie_id ON aka_title USING BTREE (movie_id);
CREATE INDEX idx_complete_cast_movie_id ON complete_cast USING BTREE (movie_id);
CREATE INDEX idx_cast_info_role_id ON cast_info USING BTREE (role_id);

LOAD DATA LOCAL INFILE './person_info.csv'
INTO TABLE person_info
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE person_info;

LOAD DATA LOCAL INFILE './info_type.csv'
INTO TABLE info_type
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE info_type;

LOAD DATA LOCAL INFILE './aka_title.csv'
INTO TABLE aka_title
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE aka_title;

LOAD DATA LOCAL INFILE './name.csv'
INTO TABLE name
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE name;

LOAD DATA LOCAL INFILE './movie_info_idx.csv'
INTO TABLE movie_info_idx
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE movie_info_idx;

LOAD DATA LOCAL INFILE './char_name.csv'
INTO TABLE char_name
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE char_name;

LOAD DATA LOCAL INFILE './company_name.csv'
INTO TABLE company_name
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE company_name;

LOAD DATA LOCAL INFILE './aka_name.csv'
INTO TABLE aka_name
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE aka_name;

LOAD DATA LOCAL INFILE './cast_info.csv'
INTO TABLE cast_info
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE cast_info;

LOAD DATA LOCAL INFILE './comp_cast_type.csv'
INTO TABLE comp_cast_type
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE comp_cast_type;

LOAD DATA LOCAL INFILE './company_type.csv'
INTO TABLE company_type
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE company_type;

LOAD DATA LOCAL INFILE './movie_link.csv'
INTO TABLE movie_link
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE movie_link;

LOAD DATA LOCAL INFILE './keyword.csv'
INTO TABLE keyword
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE keyword;

LOAD DATA LOCAL INFILE './title.csv'
INTO TABLE title
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE title;

LOAD DATA LOCAL INFILE './movie_info.csv'
INTO TABLE movie_info
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE movie_info;

LOAD DATA LOCAL INFILE './movie_keyword.csv'
INTO TABLE movie_keyword
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE movie_keyword;

LOAD DATA LOCAL INFILE './link_type.csv'
INTO TABLE link_type
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE link_type;

LOAD DATA LOCAL INFILE './complete_cast.csv'
INTO TABLE complete_cast
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE complete_cast;

LOAD DATA LOCAL INFILE './kind_type.csv'
INTO TABLE kind_type
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE kind_type;

LOAD DATA LOCAL INFILE './role_type.csv'
INTO TABLE role_type
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE role_type;

LOAD DATA LOCAL INFILE './movie_companies.csv'
INTO TABLE movie_companies
FIELDS TERMINATED BY '|'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
NULL DEFINED BY 'NULL';
ANALYZE TABLE movie_companies;
