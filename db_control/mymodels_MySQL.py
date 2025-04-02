from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Time
)
from sqlalchemy.orm import relationship, declarative_base
import datetime
import pytz

JST = pytz.timezone("Asia/Tokyo")
Base = declarative_base()

# Usersテーブル
class User(Base):
    __tablename__ = "users"
    id = Column(String(100), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    affiliation = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    total_thanks = Column(Integer, default=0)
    total_view = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    meetings = relationship("Meeting", back_populates="user")
    knowledges = relationship("Knowledge", back_populates="user")
    challenges = relationship("Challenge", back_populates="user")

# Meetingテーブル
class Meeting(Base):
    __tablename__ = "meeting"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), ForeignKey("users.id"), nullable=False)
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    # time = Column(Time, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))
    user = relationship("User", back_populates="meetings")
    knowledges = relationship("Knowledge", back_populates="meeting")
    challenges = relationship("Challenge", back_populates="meeting")

# Knowledgeテーブル
class Knowledge(Base):
    __tablename__ = "knowledge"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    meeting_id = Column(Integer, ForeignKey("meeting.id"), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    thanks_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    user = relationship("User", back_populates="knowledges")
    meeting = relationship("Meeting", back_populates="knowledges")
    tags = relationship("Tag", secondary="knowledge_tags", back_populates="knowledges")

# Challengeテーブル
class Challenge(Base):
    __tablename__ = "challenge"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    meeting_id = Column(Integer, ForeignKey("meeting.id"), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

    user = relationship("User", back_populates="challenges")
    meeting = relationship("Meeting", back_populates="challenges")
    tags = relationship("Tag", secondary="challenge_tags", back_populates="challenges")

# Tagテーブル
class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)

    knowledges = relationship("Knowledge", secondary="knowledge_tags", back_populates="tags")
    challenges = relationship("Challenge", secondary="challenge_tags", back_populates="tags")

# KnowledgeTagテーブル（中間テーブル）
class KnowledgeTag(Base):
    __tablename__ = "knowledge_tags"
    knowledge_id = Column(Integer, ForeignKey("knowledge.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)

# ChallengeTagテーブル（中間テーブル）
class ChallengeTag(Base):
    __tablename__ = "challenge_tags"
    challenge_id = Column(Integer, ForeignKey("challenge.id"), primary_key=True)
    tag_id = Column(Integer, ForeignKey("tags.id"), primary_key=True)

# Thanksテーブル
class Thanks(Base):
    __tablename__ = "thanks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    knowledge_id = Column(Integer, ForeignKey("knowledge.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

# Viewテーブル
class View(Base):
    __tablename__ = "view"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    knowledge_id = Column(Integer, ForeignKey("knowledge.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(JST))

# 以下、practicalのコード

# from sqlalchemy import String, Integer, Column, DateTime
# import datetime, pytz
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# class Base(DeclarativeBase):
    # pass

# class Customers(Base):
#     __tablename__ = 'customers'
#     customer_id: Mapped[str] = mapped_column(String(10), primary_key=True)
#     customer_name: Mapped[str] = mapped_column(String(100))
#     age: Mapped[int] = mapped_column(Integer)
#     gender: Mapped[str] = mapped_column(String(10))