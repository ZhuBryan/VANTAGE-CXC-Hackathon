# VANTAGE: Satellite AI Discrepancy Dashboard

VANTAGE is a high-performance, responsive monitoring tool designed to expose corporate environmental discrepancies. By comparing real-world satellite data against corporate sustainability claims, the platform generates real-time Hypocrisy Scores to ensure environmental accountability.

## Overview

Built during a high-intensity hackathon, VANTAGE bridges the gap between corporate PR and geospatial reality. The application processes satellite metadata logs to identify land-use changes—such as deforestation or industrial expansion—and visualizes these discrepancies through a cutting-edge Glassmorphism interface.

The project was completed by a team of three.

## Technical Stack

* Frontend: Reflex (Python), Tailwind CSS
* State Management: Reflex Reactive State
* Machine Learning: PyTorch (ResNet-18 Transfer Learning), EuroSAT Dataset
* Data Infrastructure: Snowflake API, SQL
* Design: Glassmorphism UI, CSS3 Keyframes

## Key Features

* Dual-Mode Responsive Design: A custom-engineered interface using rx.tablet_and_desktop and rx.mobile_only to provide a dense Desktop HUD and a vertically optimized Mobile Feed.
* Real-Time Analytics: Processes 1,000+ simulated satellite logs with sub-200ms latency to calculate live Hypocrisy Scores across 5+ mock global enterprises.
* Reactive Visual Alerts: Conditional styling and CSS keyframe animations that trigger visual Danger states when discrepancy thresholds exceed 60%.
* Integrated Data Pipeline: Connects Snowflake metadata queries directly to a PyTorch-based analysis backend for seamless evidence tracking.

## Architecture

VANTAGE utilizes a streamlined architecture to maintain high performance under load:

1. The Core: A unified Reflex environment where backend logic and frontend UI share a synchronized state for real-time updates.
2. Computer Vision: A PyTorch backend fine-tuned to distinguish between Forest, Industrial, and River signatures using multispectral imagery.
3. Data Warehouse: Snowflake API integration managing high-volume metadata for sub-second historical evidence retrieval.
4. Interface: A Tailwind-powered dashboard utilizing backdrop filters, custom reactive borders, and device-specific layouts.

## Development and Contributions

Following the team reduction, work was split among three members to ensure all core systems were delivered.

* Frontend and Integration: Spearheaded the Reflex and Tailwind CSS architecture, developing the dual-mode responsive layout and the reactive state management logic that bridges the ML analysis with the UI.
* ML and Data: Engineered the PyTorch training pipeline and the Snowflake API integration for metadata management.
* System Design: Collaborative implementation of the "Claim vs. Reality" logic and the corporate discrepancy scoring system.

---

Developed for CxC 2026
