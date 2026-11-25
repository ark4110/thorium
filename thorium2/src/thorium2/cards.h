// Thorium 2 — reading of NASTRAN-style bulk data cards (used by both *.bdf and *.top files).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <istream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace th {

/// Error in user input. The message always names the file and the line.
class InputError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// One logical card after joining continuation lines.
///
/// Field numbering follows NASTRAN: field(1) is the card name, fields 2..9 are the data
/// fields of the first line, a continuation line adds 8 more data fields (10..17 are
/// fields 2..9 of the second line, and so on). The continuation-marker columns (field 1
/// of a continuation line, field 10 of any line) are not data and are not stored.
struct Card {
  std::string name;               ///< Upper-case card name without trailing '*'.
  std::vector<std::string> data;  ///< Data fields, trimmed; empty string = blank field.
  std::string file;
  int line = 0;                   ///< Line of the card name (1-based).

  /// Data field by NASTRAN-style index: 2..9 = first line, 10..17 = first continuation, ...
  const std::string& field(int nastran_index) const;
  /// Data field of a given logical line: line 0 = card line, line 1 = first continuation.
  /// `field_in_line` is 2..9.
  const std::string& field(int line_index, int field_in_line) const;
  int line_count() const { return static_cast<int>((data.size() + 7) / 8); }

  bool blank(int nastran_index) const { return field(nastran_index).empty(); }

  /// Integer field. Throws InputError if blank (unless `def` given) or not an integer.
  int integer(int nastran_index, std::optional<int> def = std::nullopt) const;
  /// Real field. Accepts NASTRAN forms ("7.+10", "1.5-3", "2.D3"). An integer literal is
  /// accepted as a real too. Throws InputError if blank (unless `def` given) or malformed.
  double real(int nastran_index, std::optional<double> def = std::nullopt) const;

  /// "file:line: CARD: " prefix for error messages.
  std::string where() const;
  [[noreturn]] void fail(const std::string& message) const;
};

/// True for an integer literal ("12", "-3", "+7"): such a field in a *.top file means
/// "number of a DEPEND card", while a literal with '.' or exponent is a constant.
bool is_integer_literal(const std::string& s);
/// Parses a NASTRAN real. Returns nullopt if `s` is not a number.
std::optional<double> parse_real(const std::string& s);
std::optional<int> parse_integer(const std::string& s);

/// Reads all cards from a stream.
///
/// Supported: small field (8 columns), large field (name ending with '*', 16 columns),
/// free field (comma separated), tab characters (expanded to the next 8-column stop),
/// '$' comments, blank lines, continuation lines starting with blank field 1, '+' or '*'.
/// Lines before "BEGIN BULK" are skipped if such a line exists; reading stops at ENDDATA.
std::vector<Card> read_cards(std::istream& in, const std::string& file_name);
std::vector<Card> read_cards_file(const std::string& path);

}  // namespace th
