# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of RealEstateOptionsGym
- Four Gymnasium-compatible environments:
  - `LeaseOption-v1`: Single property lease-option trading
  - `PropertyPortfolio-v1`: Multi-property portfolio management
  - `MortgageHedging-v1`: Interest rate hedging with derivatives
  - `REITTrading-v1`: REIT trading with NAV modeling
- Stochastic interest rate models:
  - Hull-White model
  - Vasicek model
  - Cox-Ingersoll-Ross (CIR) model
- Property dynamics model with:
  - Mean reversion
  - Regime-switching volatility
  - Jump-diffusion component
- Data adapters:
  - UK Land Registry price paid data
  - FRED interest rate data
- FinRL integration wrapper
- Comprehensive test suite
- Examples for basic usage and PPO training

## [0.1.0] - 2025-XX-XX

### Added
- Initial public release

[Unreleased]: https://github.com/BrandPennington/realestate-options-gym/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BrandPennington/realestate-options-gym/releases/tag/v0.1.0
