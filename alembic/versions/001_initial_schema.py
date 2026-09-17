"""Initial Database Schema Migration for Users, Breeds, Images, Predictions, and Model Versions.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-08-27 22:00:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create Users Table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # 2. Create Breeds Table
    op.create_table(
        'breeds',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('breed_name', sa.String(length=100), nullable=False),
        sa.Column('animal_type', sa.String(length=50), nullable=False),
        sa.Column('origin', sa.String(length=255), nullable=False),
        sa.Column('native_state', sa.String(length=100), nullable=False),
        sa.Column('physical_characteristics', sa.JSON(), nullable=False),
        sa.Column('milk_production', sa.JSON(), nullable=False),
        sa.Column('climate_adaptability', sa.Text(), nullable=False),
        sa.Column('uses', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_breeds_breed_name'), 'breeds', ['breed_name'], unique=True)
    op.create_index(op.f('ix_breeds_animal_type'), 'breeds', ['animal_type'], unique=False)

    # 3. Create Images Table
    op.create_table(
        'images',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.String(length=512), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('md5_hash', sa.String(length=32), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_images_md5_hash'), 'images', ['md5_hash'], unique=False)

    # 4. Create Model Versions Table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('yolo_version', sa.String(length=50), nullable=False),
        sa.Column('efficientnet_version', sa.String(length=50), nullable=False),
        sa.Column('gradcam_version', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # 5. Create Predictions Table
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('image_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('model_version_id', sa.Integer(), nullable=False),
        sa.Column('animal_type', sa.String(length=50), nullable=False),
        sa.Column('animal_confidence', sa.Float(), nullable=False),
        sa.Column('bounding_box', sa.JSON(), nullable=False),
        sa.Column('predicted_breed_id', sa.Integer(), nullable=True),
        sa.Column('predicted_breed_name', sa.String(length=100), nullable=False),
        sa.Column('breed_confidence', sa.Float(), nullable=False),
        sa.Column('top_3_predictions', sa.JSON(), nullable=False),
        sa.Column('prediction_status', sa.String(length=50), nullable=False),
        sa.Column('inference_time_ms', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], ),
        sa.ForeignKeyConstraint(['predicted_breed_id'], ['breeds.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('predictions')
    op.drop_table('model_versions')
    op.drop_table('images')
    op.drop_table('breeds')
    op.drop_table('users')
