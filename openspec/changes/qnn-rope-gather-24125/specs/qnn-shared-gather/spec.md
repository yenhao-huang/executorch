# Spec Delta

## Purpose

Preserve numerical correctness when QNN graphs share gathered position tables
between multiple rotary embedding consumers, including output-only consumers.

## ADDED Requirements

### Requirement: Shared gathered values remain correct at graph outputs

The QNN delegate SHALL preserve gathered values for every consumer regardless
of whether that consumer also feeds another operation or only a graph output.

#### Scenario: Two quantized RoPE consumers

- **WHEN** the reported seed-zero 16a8w model gathers cosine and sine tables for two RoPEs, with only the first also feeding a batch matrix multiplication
- **THEN** both returned key tensors agree with the floating reference within a measured quantization tolerance, without periodic row substitution

#### Scenario: Unaffected controls

- **WHEN** the model uses FP16 or direct cosine/sine inputs
- **THEN** the outputs remain numerically correct within the precision-specific tolerance

#### Scenario: A single output-only RoPE

- **WHEN** one 16a8w RoPE consumes the same gathered table form and feeds only a graph output
- **THEN** the corrected lowering also agrees with the reference, without relying on multiple consumers
