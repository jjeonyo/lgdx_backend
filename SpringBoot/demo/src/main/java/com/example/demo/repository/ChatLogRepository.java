package com.example.demo.repository;

import com.example.demo.entity.ChatLog;
import org.springframework.data.jpa.repository.JpaRepository;

// JPA가 알아서 DB 저장 기능을 만들어줍니다. (마법!)
public interface ChatLogRepository extends JpaRepository<ChatLog, Long> {
}