-- MySQL translation of TPC-DS DSB schema and data load script
-- Requires local_infile=1 to be enabled on the MySQL server/client.
SET NAMES utf8mb4;

CREATE TABLE dbgen_version (
    dv_version VARCHAR(16),
    dv_create_date DATE,
    dv_create_time TIME,
    dv_cmdline_args VARCHAR(200)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE customer_address (
    ca_address_sk INT NOT NULL,
    ca_address_id CHAR(16) NOT NULL,
    ca_street_number CHAR(10),
    ca_street_name VARCHAR(60),
    ca_street_type CHAR(15),
    ca_suite_number CHAR(10),
    ca_city VARCHAR(60),
    ca_county VARCHAR(30),
    ca_state CHAR(2),
    ca_zip CHAR(10),
    ca_country VARCHAR(20),
    ca_gmt_offset DECIMAL(5,2),
    ca_location_type CHAR(20),
    PRIMARY KEY (ca_address_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE customer_demographics (
    cd_demo_sk INT NOT NULL,
    cd_gender CHAR(1),
    cd_marital_status CHAR(1),
    cd_education_status CHAR(20),
    cd_purchase_estimate INT,
    cd_credit_rating CHAR(10),
    cd_dep_count INT,
    cd_dep_employed_count INT,
    cd_dep_college_count INT,
    PRIMARY KEY (cd_demo_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE date_dim (
    d_date_sk INT NOT NULL,
    d_date_id CHAR(16) NOT NULL,
    d_date DATE,
    d_month_seq INT,
    d_week_seq INT,
    d_quarter_seq INT,
    d_year INT,
    d_dow INT,
    d_moy INT,
    d_dom INT,
    d_qoy INT,
    d_fy_year INT,
    d_fy_quarter_seq INT,
    d_fy_week_seq INT,
    d_day_name CHAR(9),
    d_quarter_name CHAR(6),
    d_holiday CHAR(1),
    d_weekend CHAR(1),
    d_following_holiday CHAR(1),
    d_first_dom INT,
    d_last_dom INT,
    d_same_day_ly INT,
    d_same_day_lq INT,
    d_current_day CHAR(1),
    d_current_week CHAR(1),
    d_current_month CHAR(1),
    d_current_quarter CHAR(1),
    d_current_year CHAR(1),
    PRIMARY KEY (d_date_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE warehouse (
    w_warehouse_sk INT NOT NULL,
    w_warehouse_id CHAR(16) NOT NULL,
    w_warehouse_name VARCHAR(20),
    w_warehouse_sq_ft INT,
    w_street_number CHAR(10),
    w_street_name VARCHAR(60),
    w_street_type CHAR(15),
    w_suite_number CHAR(10),
    w_city VARCHAR(60),
    w_county VARCHAR(30),
    w_state CHAR(2),
    w_zip CHAR(10),
    w_country VARCHAR(20),
    w_gmt_offset DECIMAL(5,2),
    PRIMARY KEY (w_warehouse_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE ship_mode (
    sm_ship_mode_sk INT NOT NULL,
    sm_ship_mode_id CHAR(16) NOT NULL,
    sm_type CHAR(30),
    sm_code CHAR(10),
    sm_carrier CHAR(20),
    sm_contract CHAR(20),
    PRIMARY KEY (sm_ship_mode_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE time_dim (
    t_time_sk INT NOT NULL,
    t_time_id CHAR(16) NOT NULL,
    t_time INT,
    t_hour INT,
    t_minute INT,
    t_second INT,
    t_am_pm CHAR(2),
    t_shift CHAR(20),
    t_sub_shift CHAR(20),
    t_meal_time CHAR(20),
    PRIMARY KEY (t_time_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE reason (
    r_reason_sk INT NOT NULL,
    r_reason_id CHAR(16) NOT NULL,
    r_reason_desc CHAR(100),
    PRIMARY KEY (r_reason_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE income_band (
    ib_income_band_sk INT NOT NULL,
    ib_lower_bound INT,
    ib_upper_bound INT,
    PRIMARY KEY (ib_income_band_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE item (
    i_item_sk INT NOT NULL,
    i_item_id CHAR(16) NOT NULL,
    i_rec_start_date DATE,
    i_rec_end_date DATE,
    i_item_desc VARCHAR(200),
    i_current_price DECIMAL(7,2),
    i_wholesale_cost DECIMAL(7,2),
    i_brand_id INT,
    i_brand CHAR(50),
    i_class_id INT,
    i_class CHAR(50),
    i_category_id INT,
    i_category CHAR(50),
    i_manufact_id INT,
    i_manufact CHAR(50),
    i_size CHAR(20),
    i_formulation CHAR(20),
    i_color CHAR(20),
    i_units CHAR(10),
    i_container CHAR(10),
    i_manager_id INT,
    i_product_name CHAR(50),
    PRIMARY KEY (i_item_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE store (
    s_store_sk INT NOT NULL,
    s_store_id CHAR(16) NOT NULL,
    s_rec_start_date DATE,
    s_rec_end_date DATE,
    s_closed_date_sk INT,
    s_store_name VARCHAR(50),
    s_number_employees INT,
    s_floor_space INT,
    s_hours CHAR(20),
    s_manager VARCHAR(40),
    s_market_id INT,
    s_geography_class VARCHAR(100),
    s_market_desc VARCHAR(100),
    s_market_manager VARCHAR(40),
    s_division_id INT,
    s_division_name VARCHAR(50),
    s_company_id INT,
    s_company_name VARCHAR(50),
    s_street_number VARCHAR(10),
    s_street_name VARCHAR(60),
    s_street_type CHAR(15),
    s_suite_number CHAR(10),
    s_city VARCHAR(60),
    s_county VARCHAR(30),
    s_state CHAR(2),
    s_zip CHAR(10),
    s_country VARCHAR(20),
    s_gmt_offset DECIMAL(5,2),
    s_tax_precentage DECIMAL(5,2),
    PRIMARY KEY (s_store_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE call_center (
    cc_call_center_sk INT NOT NULL,
    cc_call_center_id CHAR(16) NOT NULL,
    cc_rec_start_date DATE,
    cc_rec_end_date DATE,
    cc_closed_date_sk INT,
    cc_open_date_sk INT,
    cc_name VARCHAR(50),
    cc_class VARCHAR(50),
    cc_employees INT,
    cc_sq_ft INT,
    cc_hours CHAR(20),
    cc_manager VARCHAR(40),
    cc_mkt_id INT,
    cc_mkt_class CHAR(50),
    cc_mkt_desc VARCHAR(100),
    cc_market_manager VARCHAR(40),
    cc_division INT,
    cc_division_name VARCHAR(50),
    cc_company INT,
    cc_company_name CHAR(50),
    cc_street_number CHAR(10),
    cc_street_name VARCHAR(60),
    cc_street_type CHAR(15),
    cc_suite_number CHAR(10),
    cc_city VARCHAR(60),
    cc_county VARCHAR(30),
    cc_state CHAR(2),
    cc_zip CHAR(10),
    cc_country VARCHAR(20),
    cc_gmt_offset DECIMAL(5,2),
    cc_tax_percentage DECIMAL(5,2),
    PRIMARY KEY (cc_call_center_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE customer (
    c_customer_sk INT NOT NULL,
    c_customer_id CHAR(16) NOT NULL,
    c_current_cdemo_sk INT,
    c_current_hdemo_sk INT,
    c_current_addr_sk INT,
    c_first_shipto_date_sk INT,
    c_first_sales_date_sk INT,
    c_salutation CHAR(10),
    c_first_name CHAR(20),
    c_last_name CHAR(30),
    c_preferred_cust_flag CHAR(1),
    c_birth_day INT,
    c_birth_month INT,
    c_birth_year INT,
    c_birth_country VARCHAR(20),
    c_login CHAR(13),
    c_email_address CHAR(50),
    c_last_review_date_sk INT,
    PRIMARY KEY (c_customer_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE web_site (
    web_site_sk INT NOT NULL,
    web_site_id CHAR(16) NOT NULL,
    web_rec_start_date DATE,
    web_rec_end_date DATE,
    web_name VARCHAR(50),
    web_open_date_sk INT,
    web_close_date_sk INT,
    web_class VARCHAR(50),
    web_manager VARCHAR(40),
    web_mkt_id INT,
    web_mkt_class VARCHAR(50),
    web_mkt_desc VARCHAR(100),
    web_market_manager VARCHAR(40),
    web_company_id INT,
    web_company_name CHAR(50),
    web_street_number CHAR(10),
    web_street_name VARCHAR(60),
    web_street_type CHAR(15),
    web_suite_number CHAR(10),
    web_city VARCHAR(60),
    web_county VARCHAR(30),
    web_state CHAR(2),
    web_zip CHAR(10),
    web_country VARCHAR(20),
    web_gmt_offset DECIMAL(5,2),
    web_tax_percentage DECIMAL(5,2),
    PRIMARY KEY (web_site_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE store_returns (
    sr_returned_date_sk INT,
    sr_return_time_sk INT,
    sr_item_sk INT NOT NULL,
    sr_customer_sk INT,
    sr_cdemo_sk INT,
    sr_hdemo_sk INT,
    sr_addr_sk INT,
    sr_store_sk INT,
    sr_reason_sk INT,
    sr_ticket_number INT NOT NULL,
    sr_return_quantity INT,
    sr_return_amt DECIMAL(7,2),
    sr_return_tax DECIMAL(7,2),
    sr_return_amt_inc_tax DECIMAL(7,2),
    sr_fee DECIMAL(7,2),
    sr_return_ship_cost DECIMAL(7,2),
    sr_refunded_cash DECIMAL(7,2),
    sr_reversed_charge DECIMAL(7,2),
    sr_store_credit DECIMAL(7,2),
    sr_net_loss DECIMAL(7,2),
    PRIMARY KEY (sr_item_sk, sr_ticket_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE household_demographics (
    hd_demo_sk INT NOT NULL,
    hd_income_band_sk INT,
    hd_buy_potential CHAR(15),
    hd_dep_count INT,
    hd_vehicle_count INT,
    PRIMARY KEY (hd_demo_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE web_page (
    wp_web_page_sk INT NOT NULL,
    wp_web_page_id CHAR(16) NOT NULL,
    wp_rec_start_date DATE,
    wp_rec_end_date DATE,
    wp_creation_date_sk INT,
    wp_access_date_sk INT,
    wp_autogen_flag CHAR(1),
    wp_customer_sk INT,
    wp_url VARCHAR(100),
    wp_type CHAR(50),
    wp_char_count INT,
    wp_link_count INT,
    wp_image_count INT,
    wp_max_ad_count INT,
    PRIMARY KEY (wp_web_page_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE promotion (
    p_promo_sk INT NOT NULL,
    p_promo_id CHAR(16) NOT NULL,
    p_start_date_sk INT,
    p_end_date_sk INT,
    p_item_sk INT,
    p_cost DECIMAL(15,2),
    p_response_target INT,
    p_promo_name CHAR(50),
    p_channel_dmail CHAR(1),
    p_channel_email CHAR(1),
    p_channel_catalog CHAR(1),
    p_channel_tv CHAR(1),
    p_channel_radio CHAR(1),
    p_channel_press CHAR(1),
    p_channel_event CHAR(1),
    p_channel_demo CHAR(1),
    p_channel_details VARCHAR(100),
    p_purpose CHAR(15),
    p_discount_active CHAR(1),
    PRIMARY KEY (p_promo_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE catalog_page (
    cp_catalog_page_sk INT NOT NULL,
    cp_catalog_page_id CHAR(16) NOT NULL,
    cp_start_date_sk INT,
    cp_end_date_sk INT,
    cp_department VARCHAR(50),
    cp_catalog_number INT,
    cp_catalog_page_number INT,
    cp_description VARCHAR(100),
    cp_type VARCHAR(100),
    PRIMARY KEY (cp_catalog_page_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE inventory (
    inv_date_sk INT NOT NULL,
    inv_item_sk INT NOT NULL,
    inv_warehouse_sk INT NOT NULL,
    inv_quantity_on_hand INT,
    PRIMARY KEY (inv_date_sk, inv_item_sk, inv_warehouse_sk)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE catalog_returns (
    cr_returned_date_sk INT,
    cr_returned_time_sk INT,
    cr_item_sk INT NOT NULL,
    cr_refunded_customer_sk INT,
    cr_refunded_cdemo_sk INT,
    cr_refunded_hdemo_sk INT,
    cr_refunded_addr_sk INT,
    cr_returning_customer_sk INT,
    cr_returning_cdemo_sk INT,
    cr_returning_hdemo_sk INT,
    cr_returning_addr_sk INT,
    cr_call_center_sk INT,
    cr_catalog_page_sk INT,
    cr_ship_mode_sk INT,
    cr_warehouse_sk INT,
    cr_reason_sk INT,
    cr_order_number INT NOT NULL,
    cr_return_quantity INT,
    cr_return_amount DECIMAL(7,2),
    cr_return_tax DECIMAL(7,2),
    cr_return_amt_inc_tax DECIMAL(7,2),
    cr_fee DECIMAL(7,2),
    cr_return_ship_cost DECIMAL(7,2),
    cr_refunded_cash DECIMAL(7,2),
    cr_reversed_charge DECIMAL(7,2),
    cr_store_credit DECIMAL(7,2),
    cr_net_loss DECIMAL(7,2),
    PRIMARY KEY (cr_item_sk, cr_order_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE web_returns (
    wr_returned_date_sk INT,
    wr_returned_time_sk INT,
    wr_item_sk INT NOT NULL,
    wr_refunded_customer_sk INT,
    wr_refunded_cdemo_sk INT,
    wr_refunded_hdemo_sk INT,
    wr_refunded_addr_sk INT,
    wr_returning_customer_sk INT,
    wr_returning_cdemo_sk INT,
    wr_returning_hdemo_sk INT,
    wr_returning_addr_sk INT,
    wr_web_page_sk INT,
    wr_reason_sk INT,
    wr_order_number INT NOT NULL,
    wr_return_quantity INT,
    wr_return_amt DECIMAL(7,2),
    wr_return_tax DECIMAL(7,2),
    wr_return_amt_inc_tax DECIMAL(7,2),
    wr_fee DECIMAL(7,2),
    wr_return_ship_cost DECIMAL(7,2),
    wr_refunded_cash DECIMAL(7,2),
    wr_reversed_charge DECIMAL(7,2),
    wr_account_credit DECIMAL(7,2),
    wr_net_loss DECIMAL(7,2),
    PRIMARY KEY (wr_item_sk, wr_order_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE web_sales (
    ws_sold_date_sk INT,
    ws_sold_time_sk INT,
    ws_ship_date_sk INT,
    ws_item_sk INT NOT NULL,
    ws_bill_customer_sk INT,
    ws_bill_cdemo_sk INT,
    ws_bill_hdemo_sk INT,
    ws_bill_addr_sk INT,
    ws_ship_customer_sk INT,
    ws_ship_cdemo_sk INT,
    ws_ship_hdemo_sk INT,
    ws_ship_addr_sk INT,
    ws_web_page_sk INT,
    ws_web_site_sk INT,
    ws_ship_mode_sk INT,
    ws_warehouse_sk INT,
    ws_promo_sk INT,
    ws_order_number INT NOT NULL,
    ws_quantity INT,
    ws_wholesale_cost DECIMAL(7,2),
    ws_list_price DECIMAL(7,2),
    ws_sales_price DECIMAL(7,2),
    ws_ext_discount_amt DECIMAL(7,2),
    ws_ext_sales_price DECIMAL(7,2),
    ws_ext_wholesale_cost DECIMAL(7,2),
    ws_ext_list_price DECIMAL(7,2),
    ws_ext_tax DECIMAL(7,2),
    ws_coupon_amt DECIMAL(7,2),
    ws_ext_ship_cost DECIMAL(7,2),
    ws_net_paid DECIMAL(7,2),
    ws_net_paid_inc_tax DECIMAL(7,2),
    ws_net_paid_inc_ship DECIMAL(7,2),
    ws_net_paid_inc_ship_tax DECIMAL(7,2),
    ws_net_profit DECIMAL(7,2),
    PRIMARY KEY (ws_item_sk, ws_order_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE catalog_sales (
    cs_sold_date_sk INT,
    cs_sold_time_sk INT,
    cs_ship_date_sk INT,
    cs_bill_customer_sk INT,
    cs_bill_cdemo_sk INT,
    cs_bill_hdemo_sk INT,
    cs_bill_addr_sk INT,
    cs_ship_customer_sk INT,
    cs_ship_cdemo_sk INT,
    cs_ship_hdemo_sk INT,
    cs_ship_addr_sk INT,
    cs_call_center_sk INT,
    cs_catalog_page_sk INT,
    cs_ship_mode_sk INT,
    cs_warehouse_sk INT,
    cs_item_sk INT NOT NULL,
    cs_promo_sk INT,
    cs_order_number INT NOT NULL,
    cs_quantity INT,
    cs_wholesale_cost DECIMAL(7,2),
    cs_list_price DECIMAL(7,2),
    cs_sales_price DECIMAL(7,2),
    cs_ext_discount_amt DECIMAL(7,2),
    cs_ext_sales_price DECIMAL(7,2),
    cs_ext_wholesale_cost DECIMAL(7,2),
    cs_ext_list_price DECIMAL(7,2),
    cs_ext_tax DECIMAL(7,2),
    cs_coupon_amt DECIMAL(7,2),
    cs_ext_ship_cost DECIMAL(7,2),
    cs_net_paid DECIMAL(7,2),
    cs_net_paid_inc_tax DECIMAL(7,2),
    cs_net_paid_inc_ship DECIMAL(7,2),
    cs_net_paid_inc_ship_tax DECIMAL(7,2),
    cs_net_profit DECIMAL(7,2),
    PRIMARY KEY (cs_item_sk, cs_order_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE store_sales (
    ss_sold_date_sk INT,
    ss_sold_time_sk INT,
    ss_item_sk INT NOT NULL,
    ss_customer_sk INT,
    ss_cdemo_sk INT,
    ss_hdemo_sk INT,
    ss_addr_sk INT,
    ss_store_sk INT,
    ss_promo_sk INT,
    ss_ticket_number INT NOT NULL,
    ss_quantity INT,
    ss_wholesale_cost DECIMAL(7,2),
    ss_list_price DECIMAL(7,2),
    ss_sales_price DECIMAL(7,2),
    ss_ext_discount_amt DECIMAL(7,2),
    ss_ext_sales_price DECIMAL(7,2),
    ss_ext_wholesale_cost DECIMAL(7,2),
    ss_ext_list_price DECIMAL(7,2),
    ss_ext_tax DECIMAL(7,2),
    ss_coupon_amt DECIMAL(7,2),
    ss_net_paid DECIMAL(7,2),
    ss_net_paid_inc_tax DECIMAL(7,2),
    ss_net_profit DECIMAL(7,2),
    PRIMARY KEY (ss_item_sk, ss_ticket_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

LOAD DATA LOCAL INFILE './dbgen_version.dat'
INTO TABLE dbgen_version
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(dv_version, dv_create_date, dv_create_time, dv_cmdline_args);
ANALYZE TABLE dbgen_version;

LOAD DATA LOCAL INFILE './customer_address.dat'
INTO TABLE customer_address
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(ca_address_sk, ca_address_id, ca_street_number, ca_street_name, ca_street_type,
 ca_suite_number, ca_city, ca_county, ca_state, ca_zip, ca_country, ca_gmt_offset,
 ca_location_type);
ANALYZE TABLE customer_address;

LOAD DATA LOCAL INFILE './customer_demographics.dat'
INTO TABLE customer_demographics
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(cd_demo_sk, cd_gender, cd_marital_status, cd_education_status,
 cd_purchase_estimate, cd_credit_rating, cd_dep_count,
 cd_dep_employed_count, cd_dep_college_count);
ANALYZE TABLE customer_demographics;

LOAD DATA LOCAL INFILE './date_dim.dat'
INTO TABLE date_dim
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(d_date_sk, d_date_id, d_date, d_month_seq, d_week_seq, d_quarter_seq, d_year,
 d_dow, d_moy, d_dom, d_qoy, d_fy_year, d_fy_quarter_seq, d_fy_week_seq, d_day_name,
 d_quarter_name, d_holiday, d_weekend, d_following_holiday, d_first_dom, d_last_dom,
 d_same_day_ly, d_same_day_lq, d_current_day, d_current_week, d_current_month,
 d_current_quarter, d_current_year);
ANALYZE TABLE date_dim;

LOAD DATA LOCAL INFILE './warehouse.dat'
INTO TABLE warehouse
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(w_warehouse_sk, w_warehouse_id, w_warehouse_name, w_warehouse_sq_ft,
 w_street_number, w_street_name, w_street_type, w_suite_number, w_city, w_county,
 w_state, w_zip, w_country, w_gmt_offset);
ANALYZE TABLE warehouse;

LOAD DATA LOCAL INFILE './ship_mode.dat'
INTO TABLE ship_mode
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(sm_ship_mode_sk, sm_ship_mode_id, sm_type, sm_code, sm_carrier, sm_contract);
ANALYZE TABLE ship_mode;

LOAD DATA LOCAL INFILE './time_dim.dat'
INTO TABLE time_dim
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(t_time_sk, t_time_id, t_time, t_hour, t_minute, t_second, t_am_pm, t_shift,
 t_sub_shift, t_meal_time);
ANALYZE TABLE time_dim;

LOAD DATA LOCAL INFILE './reason.dat'
INTO TABLE reason
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(r_reason_sk, r_reason_id, r_reason_desc);
ANALYZE TABLE reason;

LOAD DATA LOCAL INFILE './income_band.dat'
INTO TABLE income_band
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(ib_income_band_sk, ib_lower_bound, ib_upper_bound);
ANALYZE TABLE income_band;

LOAD DATA LOCAL INFILE './item.dat'
INTO TABLE item
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(i_item_sk, i_item_id, i_rec_start_date, i_rec_end_date, i_item_desc,
 i_current_price, i_wholesale_cost, i_brand_id, i_brand, i_class_id, i_class,
 i_category_id, i_category, i_manufact_id, i_manufact, i_size, i_formulation,
 i_color, i_units, i_container, i_manager_id, i_product_name);
ANALYZE TABLE item;

LOAD DATA LOCAL INFILE './store.dat'
INTO TABLE store
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(s_store_sk, s_store_id, s_rec_start_date, s_rec_end_date, s_closed_date_sk,
 s_store_name, s_number_employees, s_floor_space, s_hours, s_manager, s_market_id,
 s_geography_class, s_market_desc, s_market_manager, s_division_id, s_division_name,
 s_company_id, s_company_name, s_street_number, s_street_name, s_street_type,
 s_suite_number, s_city, s_county, s_state, s_zip, s_country, s_gmt_offset,
 s_tax_precentage);
ANALYZE TABLE store;

LOAD DATA LOCAL INFILE './call_center.dat'
INTO TABLE call_center
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(cc_call_center_sk, cc_call_center_id, cc_rec_start_date, cc_rec_end_date,
 cc_closed_date_sk, cc_open_date_sk, cc_name, cc_class, cc_employees, cc_sq_ft,
 cc_hours, cc_manager, cc_mkt_id, cc_mkt_class, cc_mkt_desc, cc_market_manager,
 cc_division, cc_division_name, cc_company, cc_company_name, cc_street_number,
 cc_street_name, cc_street_type, cc_suite_number, cc_city, cc_county, cc_state,
 cc_zip, cc_country, cc_gmt_offset, cc_tax_percentage);
ANALYZE TABLE call_center;

LOAD DATA LOCAL INFILE './customer.dat'
INTO TABLE customer
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(c_customer_sk, c_customer_id, c_current_cdemo_sk, c_current_hdemo_sk,
 c_current_addr_sk, c_first_shipto_date_sk, c_first_sales_date_sk, c_salutation,
 c_first_name, c_last_name, c_preferred_cust_flag, c_birth_day, c_birth_month,
 c_birth_year, c_birth_country, c_login, c_email_address, c_last_review_date_sk);
ANALYZE TABLE customer;

LOAD DATA LOCAL INFILE './web_site.dat'
INTO TABLE web_site
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(web_site_sk, web_site_id, web_rec_start_date, web_rec_end_date, web_name,
 web_open_date_sk, web_close_date_sk, web_class, web_manager, web_mkt_id,
 web_mkt_class, web_mkt_desc, web_market_manager, web_company_id, web_company_name,
 web_street_number, web_street_name, web_street_type, web_suite_number, web_city,
 web_county, web_state, web_zip, web_country, web_gmt_offset, web_tax_percentage);
ANALYZE TABLE web_site;

LOAD DATA LOCAL INFILE './store_returns.dat'
INTO TABLE store_returns
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(sr_returned_date_sk, sr_return_time_sk, sr_item_sk, sr_customer_sk, sr_cdemo_sk,
 sr_hdemo_sk, sr_addr_sk, sr_store_sk, sr_reason_sk, sr_ticket_number,
 sr_return_quantity, sr_return_amt, sr_return_tax, sr_return_amt_inc_tax, sr_fee,
 sr_return_ship_cost, sr_refunded_cash, sr_reversed_charge, sr_store_credit,
 sr_net_loss);
ANALYZE TABLE store_returns;

LOAD DATA LOCAL INFILE './household_demographics.dat'
INTO TABLE household_demographics
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(hd_demo_sk, hd_income_band_sk, hd_buy_potential, hd_dep_count, hd_vehicle_count);
ANALYZE TABLE household_demographics;

LOAD DATA LOCAL INFILE './web_page.dat'
INTO TABLE web_page
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(wp_web_page_sk, wp_web_page_id, wp_rec_start_date, wp_rec_end_date,
 wp_creation_date_sk, wp_access_date_sk, wp_autogen_flag, wp_customer_sk, wp_url,
 wp_type, wp_char_count, wp_link_count, wp_image_count, wp_max_ad_count);
ANALYZE TABLE web_page;

LOAD DATA LOCAL INFILE './promotion.dat'
INTO TABLE promotion
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(p_promo_sk, p_promo_id, p_start_date_sk, p_end_date_sk, p_item_sk, p_cost,
 p_response_target, p_promo_name, p_channel_dmail, p_channel_email, p_channel_catalog,
 p_channel_tv, p_channel_radio, p_channel_press, p_channel_event, p_channel_demo,
 p_channel_details, p_purpose, p_discount_active);
ANALYZE TABLE promotion;

LOAD DATA LOCAL INFILE './catalog_page.dat'
INTO TABLE catalog_page
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(cp_catalog_page_sk, cp_catalog_page_id, cp_start_date_sk, cp_end_date_sk,
 cp_department, cp_catalog_number, cp_catalog_page_number, cp_description, cp_type);
ANALYZE TABLE catalog_page;

LOAD DATA LOCAL INFILE './inventory.dat'
INTO TABLE inventory
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(inv_date_sk, inv_item_sk, inv_warehouse_sk, inv_quantity_on_hand);
ANALYZE TABLE inventory;

LOAD DATA LOCAL INFILE './catalog_returns.dat'
INTO TABLE catalog_returns
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(cr_returned_date_sk, cr_returned_time_sk, cr_item_sk, cr_refunded_customer_sk,
 cr_refunded_cdemo_sk, cr_refunded_hdemo_sk, cr_refunded_addr_sk,
 cr_returning_customer_sk, cr_returning_cdemo_sk, cr_returning_hdemo_sk,
 cr_returning_addr_sk, cr_call_center_sk, cr_catalog_page_sk, cr_ship_mode_sk,
 cr_warehouse_sk, cr_reason_sk, cr_order_number, cr_return_quantity,
 cr_return_amount, cr_return_tax, cr_return_amt_inc_tax, cr_fee, cr_return_ship_cost,
 cr_refunded_cash, cr_reversed_charge, cr_store_credit, cr_net_loss);
ANALYZE TABLE catalog_returns;

LOAD DATA LOCAL INFILE './web_returns.dat'
INTO TABLE web_returns
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(wr_returned_date_sk, wr_returned_time_sk, wr_item_sk, wr_refunded_customer_sk,
 wr_refunded_cdemo_sk, wr_refunded_hdemo_sk, wr_refunded_addr_sk,
 wr_returning_customer_sk, wr_returning_cdemo_sk, wr_returning_hdemo_sk,
 wr_returning_addr_sk, wr_web_page_sk, wr_reason_sk, wr_order_number,
 wr_return_quantity, wr_return_amt, wr_return_tax, wr_return_amt_inc_tax, wr_fee,
 wr_return_ship_cost, wr_refunded_cash, wr_reversed_charge, wr_account_credit,
 wr_net_loss);
ANALYZE TABLE web_returns;

LOAD DATA LOCAL INFILE './web_sales.dat'
INTO TABLE web_sales
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(ws_sold_date_sk, ws_sold_time_sk, ws_ship_date_sk, ws_item_sk,
 ws_bill_customer_sk, ws_bill_cdemo_sk, ws_bill_hdemo_sk, ws_bill_addr_sk,
 ws_ship_customer_sk, ws_ship_cdemo_sk, ws_ship_hdemo_sk, ws_ship_addr_sk,
 ws_web_page_sk, ws_web_site_sk, ws_ship_mode_sk, ws_warehouse_sk, ws_promo_sk,
 ws_order_number, ws_quantity, ws_wholesale_cost, ws_list_price, ws_sales_price,
 ws_ext_discount_amt, ws_ext_sales_price, ws_ext_wholesale_cost, ws_ext_list_price,
 ws_ext_tax, ws_coupon_amt, ws_ext_ship_cost, ws_net_paid, ws_net_paid_inc_tax,
 ws_net_paid_inc_ship, ws_net_paid_inc_ship_tax, ws_net_profit);
ANALYZE TABLE web_sales;

LOAD DATA LOCAL INFILE './catalog_sales.dat'
INTO TABLE catalog_sales
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(cs_sold_date_sk, cs_sold_time_sk, cs_ship_date_sk, cs_bill_customer_sk,
 cs_bill_cdemo_sk, cs_bill_hdemo_sk, cs_bill_addr_sk, cs_ship_customer_sk,
 cs_ship_cdemo_sk, cs_ship_hdemo_sk, cs_ship_addr_sk, cs_call_center_sk,
 cs_catalog_page_sk, cs_ship_mode_sk, cs_warehouse_sk, cs_item_sk, cs_promo_sk,
 cs_order_number, cs_quantity, cs_wholesale_cost, cs_list_price, cs_sales_price,
 cs_ext_discount_amt, cs_ext_sales_price, cs_ext_wholesale_cost, cs_ext_list_price,
 cs_ext_tax, cs_coupon_amt, cs_ext_ship_cost, cs_net_paid, cs_net_paid_inc_tax,
 cs_net_paid_inc_ship, cs_net_paid_inc_ship_tax, cs_net_profit);
ANALYZE TABLE catalog_sales;

LOAD DATA LOCAL INFILE './store_sales.dat'
INTO TABLE store_sales
FIELDS TERMINATED BY '|'
LINES TERMINATED BY '\n'
(ss_sold_date_sk, ss_sold_time_sk, ss_item_sk, ss_customer_sk, ss_cdemo_sk,
 ss_hdemo_sk, ss_addr_sk, ss_store_sk, ss_promo_sk, ss_ticket_number, ss_quantity,
 ss_wholesale_cost, ss_list_price, ss_sales_price, ss_ext_discount_amt,
 ss_ext_sales_price, ss_ext_wholesale_cost, ss_ext_list_price, ss_ext_tax,
 ss_coupon_amt, ss_net_paid, ss_net_paid_inc_tax, ss_net_profit);
ANALYZE TABLE store_sales;
