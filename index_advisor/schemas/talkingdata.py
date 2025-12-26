from .schema import Column, DatabaseSchema, TableSchema


talkingdata_schema = DatabaseSchema(
    "talkingdata",
    table_count=12,
    tables=[
        TableSchema(
            "app_all",
            [
                Column(name="app_id", typ="bigint"),
            ],
            primary_key="app_id",
        ),
        TableSchema(
            "app_events",
            [
                Column(name="app_id", typ="bigint"),
                Column(name="event_id", typ="integer"),
                Column(name="is_active", typ="integer"),
                Column(name="is_installed", typ="integer"),
            ],
            primary_key=["event_id", "app_id"],
        ),
        TableSchema(
            "app_events_relevant",
            [
                Column(name="app_id", typ="bigint"),
                Column(name="event_id", typ="integer"),
                Column(name="is_active", typ="integer"),
                Column(name="is_installed", typ="integer"),
            ],
            primary_key=["event_id", "app_id"],
        ),
        TableSchema(
            "app_labels",
            [
                Column(name="app_id", typ="bigint"),
                Column(name="label_id", typ="integer"),
            ],
        ),
        TableSchema(
            "events",
            [
                Column(name="device_id", typ="bigint"),
                Column(name="event_id", typ="integer"),
                Column(name="latitude", typ="numeric"),
                Column(name="longitude", typ="numeric"),
                Column(name="timestamp", typ="timestamp without time zone"),
            ],
            primary_key="event_id",
        ),
        TableSchema(
            "events_relevant",
            [
                Column(name="device_id", typ="bigint"),
                Column(name="event_id", typ="integer"),
                Column(name="latitude", typ="numeric"),
                Column(name="longitude", typ="numeric"),
                Column(name="timestamp", typ="timestamp without time zone"),
            ],
            primary_key="event_id",
        ),
        TableSchema(
            "gender_age",
            [
                Column(name="age", typ="integer"),
                Column(name="device_id", typ="bigint"),
                Column(name="gender", typ="character varying"),
                Column(name="group1", typ="character varying"),
            ],
            primary_key="device_id",
        ),
        TableSchema(
            "gender_age_test",
            [
                Column(name="device_id", typ="bigint"),
            ],
            primary_key="device_id",
        ),
        TableSchema(
            "gender_age_train",
            [
                Column(name="age", typ="integer"),
                Column(name="device_id", typ="bigint"),
                Column(name="gender", typ="character varying"),
                Column(name="group1", typ="character varying"),
            ],
            primary_key="device_id",
        ),
        TableSchema(
            "label_categories",
            [
                Column(name="category", typ="character varying"),
                Column(name="label_id", typ="integer"),
            ],
            primary_key="label_id",
        ),
        TableSchema(
            "phone_brand_device_model2",
            [
                Column(name="device_id", typ="bigint"),
                Column(name="device_model", typ="character varying"),
                Column(name="phone_brand", typ="character varying"),
            ],
            primary_key=["device_id", "phone_brand", "device_model"],
        ),
        TableSchema(
            "sample_submission",
            [
                Column(name="device_id", typ="bigint"),
                Column(name="f23", typ="double precision"),
                Column(name="f24_26", typ="double precision"),
                Column(name="f27_28", typ="double precision"),
                Column(name="f29_32", typ="double precision"),
                Column(name="f33_42", typ="double precision"),
                Column(name="f43", typ="double precision"),
                Column(name="m22", typ="double precision"),
                Column(name="m23_26", typ="double precision"),
                Column(name="m27_28", typ="double precision"),
                Column(name="m29_31", typ="double precision"),
                Column(name="m32_38", typ="double precision"),
                Column(name="m39", typ="double precision"),
            ],
            primary_key="device_id",
        ),
    ],
    index_count=11,
    indexes=[],
)
