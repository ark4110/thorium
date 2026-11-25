// SPDX-License-Identifier: GPL-3.0-or-later
#include "thorium2/cards.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <fstream>
#include <sstream>

namespace th {

namespace {

const std::string kBlank;

std::string trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

std::string upper(std::string s) {
  for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

/// Replaces tabs by spaces up to the next multiple of 8 columns, drops '\r'.
std::string expand_tabs(const std::string& line) {
  std::string out;
  out.reserve(line.size() + 16);
  for (char c : line) {
    if (c == '\t') {
      do out.push_back(' ');
      while (out.size() % 8 != 0);
    } else if (c != '\r') {
      out.push_back(c);
    }
  }
  return out;
}

bool is_blank_line(const std::string& s) {
  return std::all_of(s.begin(), s.end(), [](char c) { return std::isspace(static_cast<unsigned char>(c)); });
}

std::string column(const std::string& line, size_t start, size_t width) {
  if (start >= line.size()) return {};
  return trim(line.substr(start, width));
}

struct Physical {
  std::string name;                 // empty for continuation lines
  std::vector<std::string> fields;  // data fields of this physical line
  bool large = false;
};

std::vector<std::string> split_commas(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  out.push_back(trim(cur));
  return out;
}

Physical split_line(const std::string& raw) {
  Physical p;
  const std::string line = expand_tabs(raw);
  const size_t first = line.find_first_not_of(' ');
  const size_t comma = line.find(',');

  // Free field: a comma inside the first 8 columns region of the first token.
  const size_t token_end = line.find_first_of(" ,", first == std::string::npos ? 0 : first);
  if (comma != std::string::npos && (token_end == std::string::npos || line[token_end] == ',')) {
    auto parts = split_commas(line);
    std::string head = upper(parts[0]);
    const bool cont = head.empty() || head[0] == '+' || head[0] == '*';
    if (!cont) {
      if (!head.empty() && head.back() == '*') head.pop_back();
      p.name = head;
    }
    for (size_t i = 1; i < parts.size() && i <= 8; ++i) p.fields.push_back(parts[i]);
    return p;
  }

  std::string head = upper(column(line, 0, 8));
  if (!head.empty() && head[0] == '*') {
    p.large = true;  // large-field continuation
  } else if (!head.empty() && head[0] != '+') {
    if (head.back() == '*') {
      p.large = true;
      head.pop_back();
    }
    p.name = head;
  }
  if (p.large) {
    for (size_t i = 0; i < 4; ++i) p.fields.push_back(column(line, 8 + 16 * i, 16));
  } else {
    for (size_t i = 0; i < 8; ++i) p.fields.push_back(column(line, 8 + 8 * i, 8));
  }
  return p;
}

}  // namespace

const std::string& Card::field(int nastran_index) const {
  if (nastran_index < 2) fail("internal error: field index " + std::to_string(nastran_index));
  const size_t i = static_cast<size_t>(nastran_index - 2);
  return i < data.size() ? data[i] : kBlank;
}

const std::string& Card::field(int line_index, int field_in_line) const {
  if (field_in_line < 2 || field_in_line > 9) fail("internal error: field " + std::to_string(field_in_line));
  return field(2 + 8 * line_index + (field_in_line - 2));
}

std::string Card::where() const {
  return file + ":" + std::to_string(line) + ": " + name + ": ";
}

void Card::fail(const std::string& message) const { throw InputError(where() + message); }

int Card::integer(int idx, std::optional<int> def) const {
  const std::string& s = field(idx);
  if (s.empty()) {
    if (def) return *def;
    fail("field " + std::to_string(idx) + " is blank, an integer is required");
  }
  auto v = parse_integer(s);
  if (!v) fail("field " + std::to_string(idx) + " = '" + s + "' is not an integer");
  return *v;
}

double Card::real(int idx, std::optional<double> def) const {
  const std::string& s = field(idx);
  if (s.empty()) {
    if (def) return *def;
    fail("field " + std::to_string(idx) + " is blank, a number is required");
  }
  auto v = parse_real(s);
  if (!v) fail("field " + std::to_string(idx) + " = '" + s + "' is not a number");
  return *v;
}

bool is_integer_literal(const std::string& s) { return parse_integer(s).has_value(); }

std::optional<int> parse_integer(const std::string& s) {
  if (s.empty()) return std::nullopt;
  size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
  if (i == s.size()) return std::nullopt;
  for (size_t k = i; k < s.size(); ++k)
    if (!std::isdigit(static_cast<unsigned char>(s[k]))) return std::nullopt;
  int value = 0;
  const char* b = s.data() + (s[0] == '+' ? 1 : 0);
  auto [ptr, ec] = std::from_chars(b, s.data() + s.size(), value);
  if (ec != std::errc() || ptr != s.data() + s.size()) return std::nullopt;
  return value;
}

std::optional<double> parse_real(const std::string& text) {
  if (text.empty()) return std::nullopt;
  std::string s = upper(text);
  std::replace(s.begin(), s.end(), 'D', 'E');
  // NASTRAN implied exponent: a sign after the mantissa, e.g. "7.+10", "1.5-3".
  if (s.find('E') == std::string::npos) {
    for (size_t i = 1; i < s.size(); ++i) {
      if ((s[i] == '+' || s[i] == '-') && (std::isdigit(static_cast<unsigned char>(s[i - 1])) || s[i - 1] == '.')) {
        s.insert(i, "E");
        break;
      }
    }
  }
  // Validate the grammar strictly: [sign] digits [. digits] [E [sign] digits].
  size_t i = 0;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
  size_t digits = 0;
  while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i, ++digits;
  if (i < s.size() && s[i] == '.') {
    ++i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i, ++digits;
  }
  if (digits == 0) return std::nullopt;
  if (i < s.size() && s[i] == 'E') {
    ++i;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
    size_t exp_digits = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i, ++exp_digits;
    if (exp_digits == 0) return std::nullopt;
  }
  if (i != s.size()) return std::nullopt;
  double value = 0;
  const char* b = s.data() + (s[0] == '+' ? 1 : 0);
  auto [ptr, ec] = std::from_chars(b, s.data() + s.size(), value);
  if (ec != std::errc() || ptr != s.data() + s.size() || !std::isfinite(value)) return std::nullopt;
  return value;
}

std::vector<Card> read_cards(std::istream& in, const std::string& file_name) {
  std::vector<std::string> lines;
  for (std::string l; std::getline(in, l);) lines.push_back(l);

  // Skip the executive/case control section if the file has BEGIN BULK.
  size_t start = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    const std::string t = upper(trim(lines[i]));
    if (t.rfind("BEGIN BULK", 0) == 0 || t.rfind("BEGIN,BULK", 0) == 0) {
      start = i + 1;
      break;
    }
  }

  std::vector<Card> cards;
  bool large_pending = false;  // large-field card waiting for its second half-line
  for (size_t i = start; i < lines.size(); ++i) {
    const std::string& raw = lines[i];
    if (raw.empty() || raw[0] == '$' || is_blank_line(expand_tabs(raw))) continue;
    const int line_no = static_cast<int>(i + 1);
    Physical p = split_line(raw);
    if (!p.name.empty()) {
      if (p.name == "ENDDATA") break;
      Card c;
      c.name = p.name;
      c.file = file_name;
      c.line = line_no;
      c.data = std::move(p.fields);
      large_pending = p.large;
      cards.push_back(std::move(c));
      continue;
    }
    if (cards.empty())
      throw InputError(file_name + ":" + std::to_string(line_no) + ": continuation line without a card");
    Card& c = cards.back();
    if (p.large && large_pending) {
      // Second half of a large-field logical line: 4 more fields complete fields 6..9.
      c.data.insert(c.data.end(), p.fields.begin(), p.fields.end());
      large_pending = false;
    } else {
      // Pad the previous logical line to 8 data fields before appending.
      c.data.resize(((c.data.size() + 7) / 8) * 8);
      c.data.insert(c.data.end(), p.fields.begin(), p.fields.end());
      large_pending = p.large;
    }
  }
  return cards;
}

std::vector<Card> read_cards_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw InputError(path + ": cannot open file");
  return read_cards(f, path);
}

}  // namespace th
