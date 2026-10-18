#include "SvgShape.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <unordered_map>

namespace {

constexpr float kPi = 3.14159265358979323846f;

std::string trimCopy(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string lowerCopy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Attribute value of `name` inside a tag body (name must be lowercase).
std::string attr(const std::string& tag, const char* name) {
    const size_t n = std::strlen(name);
    size_t pos = 0;
    while ((pos = tag.find(name, pos)) != std::string::npos) {
        const bool leftOk = (pos == 0) || !(std::isalnum(static_cast<unsigned char>(tag[pos - 1])) ||
                                            tag[pos - 1] == '-' || tag[pos - 1] == '_');
        const size_t after = pos + n;
        if (leftOk && after < tag.size()) {
            size_t eq = after;
            while (eq < tag.size() && std::isspace(static_cast<unsigned char>(tag[eq]))) ++eq;
            if (eq < tag.size() && tag[eq] == '=') {
                ++eq;
                while (eq < tag.size() && std::isspace(static_cast<unsigned char>(tag[eq]))) ++eq;
                if (eq < tag.size() && (tag[eq] == '"' || tag[eq] == '\'')) {
                    const char quote = tag[eq++];
                    const size_t end = tag.find(quote, eq);
                    if (end != std::string::npos) return tag.substr(eq, end - eq);
                } else {
                    size_t end = eq;
                    while (end < tag.size() && !std::isspace(static_cast<unsigned char>(tag[end])) &&
                           tag[end] != '/' && tag[end] != '>') {
                        ++end;
                    }
                    return tag.substr(eq, end - eq);
                }
            }
        }
        pos = after;
    }
    return {};
}

// style="fill:#abc;..." property lookup (used when the attribute is absent).
std::string styleProp(const std::string& tag, const std::string& prop) {
    std::string style = lowerCopy(attr(tag, "style"));
    if (style.empty()) return {};
    const std::string needle = prop + ":";
    size_t pos = style.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    const size_t end = style.find(';', pos);
    return trimCopy(style.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
}

std::string fillSpec(const std::string& tag) {
    std::string fill = attr(tag, "fill");
    if (fill.empty()) fill = styleProp(tag, "fill");
    return fill;
}

// 2x3 affine transform: x' = a*x + c*y + e, y' = b*x + d*y + f.
struct Affine {
    float a = 1.0f, b = 0.0f, c = 0.0f, d = 1.0f, e = 0.0f, f = 0.0f;

    glm::vec2 apply(const glm::vec2& p) const {
        return glm::vec2(a * p.x + c * p.y + e, b * p.x + d * p.y + f);
    }
    // Returns this∘inner: `inner` is applied first, then `this`.
    Affine compose(const Affine& inner) const {
        Affine out;
        out.a = a * inner.a + c * inner.b;
        out.b = b * inner.a + d * inner.b;
        out.c = a * inner.c + c * inner.d;
        out.d = b * inner.c + d * inner.d;
        out.e = a * inner.e + c * inner.f + e;
        out.f = b * inner.e + d * inner.f + f;
        return out;
    }
};

struct NumScanner {
    const char* p = nullptr;
    const char* end = nullptr;

    explicit NumScanner(const std::string& text, size_t from = 0)
        : p(text.data() + std::min(from, text.size())), end(text.data() + text.size()) {}

    void skipSeparators() {
        while (p < end && (std::isspace(static_cast<unsigned char>(*p)) || *p == ',')) ++p;
    }
    bool eof() {
        skipSeparators();
        return p >= end;
    }
    bool peekCommand(char& out) {
        skipSeparators();
        if (p < end && std::isalpha(static_cast<unsigned char>(*p))) {
            out = *p;
            return true;
        }
        return false;
    }
    bool number(float& out) {
        skipSeparators();
        if (p >= end) return false;
        char* stop = nullptr;
        const double v = std::strtod(p, &stop);
        if (stop == p) return false;
        p = stop;
        out = static_cast<float>(v);
        return true;
    }
    bool flag(bool& out) {
        skipSeparators();
        if (p >= end) return false;
        if (*p == '0' || *p == '1') {
            out = (*p == '1');
            ++p;
            return true;
        }
        float v = 0.0f;
        if (number(v)) {
            out = v != 0.0f;
            return true;
        }
        return false;
    }
};

Affine parseTransform(const std::string& text) {
    Affine result;
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() &&
               (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == ',')) {
            ++pos;
        }
        const size_t nameStart = pos;
        while (pos < text.size() && std::isalpha(static_cast<unsigned char>(text[pos]))) ++pos;
        if (pos == nameStart) break;
        const std::string name = lowerCopy(text.substr(nameStart, pos - nameStart));
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        if (pos >= text.size() || text[pos] != '(') break;
        const size_t close = text.find(')', pos);
        if (close == std::string::npos) break;

        std::string args = text.substr(pos + 1, close - pos - 1);
        NumScanner sc(args);
        float v[6] = {0, 0, 0, 0, 0, 0};
        int n = 0;
        while (n < 6 && sc.number(v[n])) ++n;

        Affine local;
        if (name == "translate" && n >= 1) {
            local.e = v[0];
            local.f = (n > 1) ? v[1] : 0.0f;
        } else if (name == "scale" && n >= 1) {
            local.a = v[0];
            local.d = (n > 1) ? v[1] : v[0];
        } else if (name == "rotate" && n >= 1) {
            const float rad = v[0] * kPi / 180.0f;
            const float cs = std::cos(rad), sn = std::sin(rad);
            Affine rot;
            rot.a = cs; rot.b = sn; rot.c = -sn; rot.d = cs;
            if (n >= 3) {
                Affine to;   to.e = v[1];    to.f = v[2];
                Affine back; back.e = -v[1]; back.f = -v[2];
                local = to.compose(rot).compose(back);
            } else {
                local = rot;
            }
        } else if (name == "matrix" && n >= 6) {
            local.a = v[0]; local.b = v[1]; local.c = v[2];
            local.d = v[3]; local.e = v[4]; local.f = v[5];
        }
        result = result.compose(local);
        pos = close + 1;
    }
    return result;
}

// Flattens SVG path data into subpaths, each emitted as (points, closed).
class PathFlattener {
public:
    using Emit = std::function<void(std::vector<glm::vec2>&&, bool)>;

    explicit PathFlattener(Emit emit) : m_emit(std::move(emit)) {}

    void run(const std::string& d) {
        NumScanner sc(d);
        char cmd = 0;
        glm::vec2 cur(0.0f), subpathStart(0.0f);
        glm::vec2 lastCubic(0.0f), lastQuad(0.0f);
        bool haveCubic = false, haveQuad = false;
        bool inSubpath = false;

        auto beginSubpath = [&](const glm::vec2& p) {
            flush(false);
            cur = subpathStart = p;
            m_points.push_back(p);
            inSubpath = true;
            haveCubic = haveQuad = false;
        };
        auto lineTo = [&](const glm::vec2& p) {
            if (!inSubpath) beginSubpath(cur);
            cur = p;
            pushPoint(p);
            haveCubic = haveQuad = false;
        };
        auto cubicTo = [&](const glm::vec2& p1, const glm::vec2& p2, const glm::vec2& p3) {
            if (!inSubpath) beginSubpath(cur);
            constexpr int kSteps = 16;
            const glm::vec2 p0 = cur;
            for (int i = 1; i <= kSteps; ++i) {
                const float t = static_cast<float>(i) / kSteps;
                const float u = 1.0f - t;
                pushPoint(u * u * u * p0 + 3.0f * u * u * t * p1 + 3.0f * u * t * t * p2 +
                          t * t * t * p3);
            }
            cur = p3;
            lastCubic = p2;
            haveCubic = true;
            haveQuad = false;
        };
        auto quadTo = [&](const glm::vec2& p1, const glm::vec2& p2) {
            if (!inSubpath) beginSubpath(cur);
            constexpr int kSteps = 12;
            const glm::vec2 p0 = cur;
            for (int i = 1; i <= kSteps; ++i) {
                const float t = static_cast<float>(i) / kSteps;
                const float u = 1.0f - t;
                pushPoint(u * u * p0 + 2.0f * u * t * p1 + t * t * p2);
            }
            cur = p2;
            lastQuad = p1;
            haveQuad = true;
            haveCubic = false;
        };

        const auto arcTo = [&](float rx, float ry, float rotationDeg, bool largeArc,
                               bool sweep, const glm::vec2& end) {
            if (!inSubpath) beginSubpath(cur);
            const glm::vec2 start = cur;
        rx = std::fabs(rx);
        ry = std::fabs(ry);
        if (rx < 1e-6f || ry < 1e-6f || glm::length(end - start) < 1e-6f) {
            lineTo(end);
            return;
        }
        const float phi = rotationDeg * kPi / 180.0f;
        const float cp = std::cos(phi), sp = std::sin(phi);

        const glm::vec2 halfDelta = (start - end) * 0.5f;
        const float x1p = cp * halfDelta.x + sp * halfDelta.y;
        const float y1p = -sp * halfDelta.x + cp * halfDelta.y;

        const float lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
        if (lambda > 1.0f) {
            const float s = std::sqrt(lambda);
            rx *= s;
            ry *= s;
        }

        const float sign = (largeArc != sweep) ? 1.0f : -1.0f;
        float num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
        const float den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
        num = std::max(0.0f, num);
        const float co = (den > 1e-12f) ? sign * std::sqrt(num / den) : 0.0f;
        const float cxp = co * (rx * y1p / ry);
        const float cyp = co * (-ry * x1p / rx);

        const float cxw = cp * cxp - sp * cyp + (start.x + end.x) * 0.5f;
        const float cyw = sp * cxp + cp * cyp + (start.y + end.y) * 0.5f;

        const auto angle = [](float ux, float uy, float vx, float vy) {
            const float dot = ux * vx + uy * vy;
            const float len = std::sqrt(std::max(1e-12f, (ux * ux + uy * uy) * (vx * vx + vy * vy)));
            const float ang = std::acos(std::clamp(dot / len, -1.0f, 1.0f));
            return (ux * vy - uy * vx < 0.0f) ? -ang : ang;
        };
        const float ux = (x1p - cxp) / rx;
        const float uy = (y1p - cyp) / ry;
        const float vx = (-x1p - cxp) / rx;
        const float vy = (-y1p - cyp) / ry;
        const float theta1 = angle(1.0f, 0.0f, ux, uy);
        float dtheta = angle(ux, uy, vx, vy);
        if (!sweep && dtheta > 0.0f) dtheta -= 2.0f * kPi;
        if (sweep && dtheta < 0.0f) dtheta += 2.0f * kPi;

        constexpr int kSteps = 24;
        for (int i = 1; i <= kSteps; ++i) {
            const float t = theta1 + dtheta * (static_cast<float>(i) / kSteps);
            const float px = cxw + rx * std::cos(t) * cp - ry * std::sin(t) * sp;
            const float py = cyw + rx * std::cos(t) * sp + ry * std::sin(t) * cp;
            pushPoint(glm::vec2(px, py));
        }
            cur = end;
        };

        while (!sc.eof()) {
            char letter = 0;
            if (sc.peekCommand(letter)) {
                cmd = letter;
                ++sc.p;
            } else if (cmd == 0) {
                break;
            }

            const bool rel = std::islower(static_cast<unsigned char>(cmd));
            const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(cmd)));
            float a = 0, b = 0, cx = 0, cy = 0, ex = 0, ey = 0;

            switch (c) {
                case 'M':
                    if (!sc.number(a) || !sc.number(b)) return;
                    beginSubpath(rel ? cur + glm::vec2(a, b) : glm::vec2(a, b));
                    cmd = rel ? 'l' : 'L';
                    break;
                case 'L':
                    if (!sc.number(a) || !sc.number(b)) return;
                    lineTo(rel ? cur + glm::vec2(a, b) : glm::vec2(a, b));
                    break;
                case 'H':
                    if (!sc.number(a)) return;
                    lineTo(rel ? glm::vec2(cur.x + a, cur.y) : glm::vec2(a, cur.y));
                    break;
                case 'V':
                    if (!sc.number(a)) return;
                    lineTo(rel ? glm::vec2(cur.x, cur.y + a) : glm::vec2(cur.x, a));
                    break;
                case 'C':
                    if (!sc.number(a) || !sc.number(b) || !sc.number(cx) || !sc.number(cy) ||
                        !sc.number(ex) || !sc.number(ey)) {
                        return;
                    }
                    cubicTo(rel ? cur + glm::vec2(a, b) : glm::vec2(a, b),
                            rel ? cur + glm::vec2(cx, cy) : glm::vec2(cx, cy),
                            rel ? cur + glm::vec2(ex, ey) : glm::vec2(ex, ey));
                    break;
                case 'S':
                    if (!sc.number(a) || !sc.number(b) || !sc.number(cx) || !sc.number(cy)) return;
                    {
                        const glm::vec2 control2 = rel ? cur + glm::vec2(a, b) : glm::vec2(a, b);
                        const glm::vec2 end = rel ? cur + glm::vec2(cx, cy) : glm::vec2(cx, cy);
                        const glm::vec2 control1 = haveCubic ? (cur * 2.0f - lastCubic) : cur;
                        cubicTo(control1, control2, end);
                    }
                    break;
                case 'Q':
                    if (!sc.number(a) || !sc.number(b) || !sc.number(cx) || !sc.number(cy)) return;
                    quadTo(rel ? cur + glm::vec2(a, b) : glm::vec2(a, b),
                           rel ? cur + glm::vec2(cx, cy) : glm::vec2(cx, cy));
                    break;
                case 'T':
                    if (!sc.number(a) || !sc.number(b)) return;
                    {
                        const glm::vec2 end = rel ? cur + glm::vec2(a, b) : glm::vec2(a, b);
                        const glm::vec2 control = haveQuad ? (cur * 2.0f - lastQuad) : cur;
                        quadTo(control, end);
                    }
                    break;
                case 'A':
                    if (!sc.number(a) || !sc.number(b) || !sc.number(cx) || !sc.number(cy)) return;
                    {
                        bool largeArc = false, sweep = false;
                        if (!sc.flag(largeArc) || !sc.flag(sweep)) return;
                        if (!sc.number(ex) || !sc.number(ey)) return;
                        arcTo(a, b, cx, largeArc, sweep,
                              rel ? cur + glm::vec2(ex, ey) : glm::vec2(ex, ey));
                    }
                    break;
                case 'Z':
                    if (inSubpath) {
                        flush(true);
                        cur = subpathStart;
                        inSubpath = false;
                    }
                    haveCubic = haveQuad = false;
                    break;
                default:
                    return;
            }
        }
        if (inSubpath) flush(false);
    }

private:
    void pushPoint(const glm::vec2& p) {
        if (!m_points.empty() && glm::length(p - m_points.back()) < 1e-6f) return;
        m_points.push_back(p);
    }

    void flush(bool closed) {
        if (m_points.size() >= 2) {
            if (closed && m_points.size() >= 3 &&
                glm::length(m_points.front() - m_points.back()) < 1e-5f) {
                m_points.pop_back();
            }
            m_emit(std::move(m_points), closed);
        }
        m_points.clear();
    }

    Emit m_emit;
    std::vector<glm::vec2> m_points;
};

float firstNumber(const std::string& text, float fallback = 0.0f) {
    NumScanner sc(text);
    float v = fallback;
    return sc.number(v) ? v : fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// SvgShape
// ---------------------------------------------------------------------------

bool SvgShape::parseColor(const std::string& text, glm::vec3& rgb) {
    const std::string s = lowerCopy(trimCopy(text));
    if (s.empty()) return false;
    if (s == "none" || s == "transparent" || s == "currentcolor") return false;
    if (s[0] == '#') {
        std::string hex = s.substr(1);
        if (hex.size() == 3) hex = std::string() + hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
        if (hex.size() != 6) return false;
        const auto digit = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            return -1;
        };
        int v[6];
        for (int i = 0; i < 6; ++i) {
            v[i] = digit(hex[i]);
            if (v[i] < 0) return false;
        }
        rgb = glm::vec3((v[0] * 16 + v[1]) / 255.0f, (v[2] * 16 + v[3]) / 255.0f,
                        (v[4] * 16 + v[5]) / 255.0f);
        return true;
    }
    if (s.rfind("rgb", 0) == 0) {
        const size_t open = s.find('(');
        const size_t close = s.find(')');
        if (open == std::string::npos || close == std::string::npos || close <= open) return false;
        const std::string components = s.substr(open + 1, close - open - 1);
        NumScanner sc(components);
        float r = 0, g = 0, b = 0;
        if (!sc.number(r) || !sc.number(g) || !sc.number(b)) return false;
        const float scale = (r > 1.0f || g > 1.0f || b > 1.0f) ? (1.0f / 255.0f) : 1.0f;
        rgb = glm::clamp(glm::vec3(r, g, b) * scale, 0.0f, 1.0f);
        return true;
    }
    static const std::unordered_map<std::string, glm::vec3> kNamed = {
        {"white", {1, 1, 1}},       {"black", {0, 0, 0}},
        {"red", {1, 0, 0}},         {"green", {0, 0.5019608f, 0}},
        {"lime", {0, 1, 0}},        {"blue", {0, 0, 1}},
        {"yellow", {1, 1, 0}},      {"cyan", {0, 1, 1}},
        {"aqua", {0, 1, 1}},        {"magenta", {1, 0, 1}},
        {"fuchsia", {1, 0, 1}},     {"gray", {0.502f, 0.502f, 0.502f}},
        {"grey", {0.502f, 0.502f, 0.502f}}, {"silver", {0.753f, 0.753f, 0.753f}},
        {"maroon", {0.502f, 0, 0}}, {"olive", {0.502f, 0.502f, 0}},
        {"navy", {0, 0, 0.502f}},   {"teal", {0, 0.502f, 0.502f}},
        {"purple", {0.502f, 0, 0.502f}}, {"orange", {1, 0.647f, 0}},
    };
    const auto it = kNamed.find(s);
    if (it != kNamed.end()) {
        rgb = it->second;
        return true;
    }
    return false;
}

glm::vec3 SvgShape::rgbToHsvDegrees(const glm::vec3& rgb) {
    const float maxC = std::max(rgb.r, std::max(rgb.g, rgb.b));
    const float minC = std::min(rgb.r, std::min(rgb.g, rgb.b));
    const float delta = maxC - minC;
    float h = 0.0f;
    if (delta > 1e-6f) {
        if (maxC == rgb.r) h = 60.0f * std::fmod((rgb.g - rgb.b) / delta, 6.0f);
        else if (maxC == rgb.g) h = 60.0f * (((rgb.b - rgb.r) / delta) + 2.0f);
        else h = 60.0f * (((rgb.r - rgb.g) / delta) + 4.0f);
    }
    if (h < 0.0f) h += 360.0f;
    const float s = (maxC <= 1e-6f) ? 0.0f : delta / maxC;
    return glm::vec3(h, s, maxC);
}

bool SvgShape::contains(const glm::vec2& p) const {
    return containingContour(p) >= 0;
}

int SvgShape::containingContour(const glm::vec2& p) const {
    int found = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        const SvgContour& contour = contours[i];
        if (!contour.closed || contour.points.size() < 3) continue;
        bool cross = false;
        for (size_t k = 0, j = contour.points.size() - 1; k < contour.points.size(); j = k++) {
            const glm::vec2& a = contour.points[k];
            const glm::vec2& b = contour.points[j];
            if (((a.y > p.y) != (b.y > p.y)) &&
                (p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)) {
                cross = !cross;
            }
        }
        if (cross) found = static_cast<int>(i);  // topmost (last drawn) wins
    }
    return found;
}

bool SvgShape::loadFromFile(const std::string& path, SvgShape& out, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open '" + path + "'";
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!fromString(text, out, error)) return false;
    out.source = path;
    return true;
}

bool SvgShape::loadFromFileResolved(const std::string& path, SvgShape& out, std::string& error,
                                    std::string* resolvedPath) {
    static const char* kPrefixes[] = { "", "bin/", "../" };
    std::string lastError = "cannot open '" + path + "'";
    for (const char* prefix : kPrefixes) {
        const std::string candidate = std::string(prefix) + path;
        if (loadFromFile(candidate, out, lastError)) {
            if (resolvedPath) *resolvedPath = candidate;
            return true;
        }
    }
    error = lastError;
    return false;
}

bool SvgShape::fromString(const std::string& rawText, SvgShape& out, std::string& error) {
    out = SvgShape();
    out.source = "<string>";
    std::string text = rawText;

    const auto stripBetween = [&text](const char* open, const char* close) {
        for (size_t pos = text.find(open); pos != std::string::npos; pos = text.find(open, pos)) {
            const size_t end = text.find(close, pos + std::strlen(open));
            if (end == std::string::npos) {
                text.erase(pos);
                break;
            }
            text.erase(pos, end + std::strlen(close) - pos);
        }
    };
    stripBetween("<!--", "-->");
    stripBetween("<?", "?>");
    stripBetween("<![CDATA[", "]]>");
    stripBetween("<!DOCTYPE", ">");

    const size_t svgPos = text.find("<svg");
    if (svgPos == std::string::npos) {
        error = "no <svg> element";
        return false;
    }
    const size_t svgEnd = text.find('>', svgPos);
    const std::string svgTag =
        text.substr(svgPos, (svgEnd == std::string::npos ? text.size() : svgEnd) - svgPos);

    // viewBox (fallback: width/height) in user units.
    {
        const std::string viewBoxText = attr(svgTag, "viewBox");
        NumScanner sc(viewBoxText);
        float vx = 0, vy = 0, vw = 0, vh = 0;
        if (sc.number(vx) && sc.number(vy) && sc.number(vw) && sc.number(vh) && vw > 1e-6f &&
            vh > 1e-6f) {
            out.viewMin = glm::vec2(vx, vy);
            out.viewSize = glm::vec2(vw, vh);
        } else {
            const std::string widthText = attr(svgTag, "width");
            const std::string heightText = attr(svgTag, "height");
            const float w = firstNumber(widthText, 0.0f);
            const float h = firstNumber(heightText, 0.0f);
            if (w > 1e-6f && h > 1e-6f) out.viewSize = glm::vec2(w, h);
        }
    }

    std::vector<Affine> transformStack{Affine()};
    std::vector<glm::vec3> fillStack{glm::vec3(1.0f)};

    const auto normalize = [&out](const glm::vec2& p) {
        return glm::vec2(2.0f * (p.x - out.viewMin.x) / out.viewSize.x - 1.0f,
                         1.0f - 2.0f * (p.y - out.viewMin.y) / out.viewSize.y);
    };
    const auto pushContour = [&out, &normalize](std::vector<glm::vec2>&& pts, bool closed,
                                                const glm::vec3& rgb) {
        if (pts.size() < (closed ? 3u : 2u)) return;
        SvgContour contour;
        contour.rgb = rgb;
        contour.closed = closed;
        contour.points.reserve(pts.size());
        for (const glm::vec2& p : pts) contour.points.push_back(normalize(p));
        out.contours.push_back(std::move(contour));
    };

    size_t pos = (svgEnd == std::string::npos) ? text.size() : svgEnd + 1;
    while (pos < text.size()) {
        const size_t lt = text.find('<', pos);
        if (lt == std::string::npos) break;
        const size_t gt = text.find('>', lt);
        if (gt == std::string::npos) break;
        const std::string trimmed = trimCopy(text.substr(lt + 1, gt - lt - 1));
        pos = gt + 1;
        if (trimmed.empty()) continue;

        if (trimmed[0] == '/') {
            if (trimmed.rfind("/g", 0) == 0) {
                if (transformStack.size() > 1) transformStack.pop_back();
                if (fillStack.size() > 1) fillStack.pop_back();
            }
            continue;
        }
        if (trimmed[0] == '!' || trimmed[0] == '?') continue;

        const size_t nameEnd = trimmed.find_first_of(" \t\r\n/");
        const std::string name = lowerCopy(trimmed.substr(0, nameEnd));
        if (name == "g") {
            const Affine xf =
                transformStack.back().compose(parseTransform(attr(trimmed, "transform")));
            transformStack.push_back(xf);
            glm::vec3 fill = fillStack.back();
            glm::vec3 parsed;
            const std::string spec = fillSpec(trimmed);
            if (!spec.empty() && parseColor(spec, parsed)) fill = parsed;
            fillStack.push_back(fill);
            continue;
        }
        if (name != "path" && name != "rect" && name != "circle" && name != "ellipse" &&
            name != "polygon" && name != "polyline" && name != "line") {
            continue;
        }

        const Affine xf =
            transformStack.back().compose(parseTransform(attr(trimmed, "transform")));
        glm::vec3 rgb = fillStack.back();
        {
            glm::vec3 parsed;
            const std::string spec = fillSpec(trimmed);
            if (!spec.empty() && parseColor(spec, parsed)) rgb = parsed;
        }

        if (name == "path") {
            PathFlattener flattener([&](std::vector<glm::vec2>&& pts, bool closed) {
                std::vector<glm::vec2> transformed;
                transformed.reserve(pts.size());
                for (const glm::vec2& p : pts) transformed.push_back(xf.apply(p));
                pushContour(std::move(transformed), closed, rgb);
            });
            flattener.run(attr(trimmed, "d"));
            continue;
        }
        if (name == "rect") {
            const float x = firstNumber(attr(trimmed, "x"), 0.0f);
            const float y = firstNumber(attr(trimmed, "y"), 0.0f);
            const float w = firstNumber(attr(trimmed, "width"), 0.0f);
            const float h = firstNumber(attr(trimmed, "height"), 0.0f);
            if (w > 0.0f && h > 0.0f) {
                std::vector<glm::vec2> pts = {xf.apply({x, y}), xf.apply({x + w, y}),
                                              xf.apply({x + w, y + h}), xf.apply({x, y + h})};
                pushContour(std::move(pts), true, rgb);
            }
            continue;
        }
        if (name == "circle" || name == "ellipse") {
            const float cx = firstNumber(attr(trimmed, "cx"), 0.0f);
            const float cy = firstNumber(attr(trimmed, "cy"), 0.0f);
            float rx = 0.0f, ry = 0.0f;
            if (name == "circle") {
                rx = ry = firstNumber(attr(trimmed, "r"), 0.0f);
            } else {
                rx = firstNumber(attr(trimmed, "rx"), 0.0f);
                ry = firstNumber(attr(trimmed, "ry"), 0.0f);
            }
            if (rx > 0.0f && ry > 0.0f) {
                constexpr int kSegments = 48;
                std::vector<glm::vec2> pts;
                pts.reserve(kSegments);
                for (int i = 0; i < kSegments; ++i) {
                    const float t = 2.0f * kPi * static_cast<float>(i) / kSegments;
                    pts.push_back(xf.apply({cx + rx * std::cos(t), cy + ry * std::sin(t)}));
                }
                pushContour(std::move(pts), true, rgb);
            }
            continue;
        }
        if (name == "polygon" || name == "polyline") {
            const std::string pointsText = attr(trimmed, "points");
            NumScanner sc(pointsText);
            std::vector<glm::vec2> pts;
            float x = 0, y = 0;
            while (sc.number(x) && sc.number(y)) pts.push_back(xf.apply({x, y}));
            pushContour(std::move(pts), name == "polygon", rgb);
            continue;
        }
        if (name == "line") {
            const float x1 = firstNumber(attr(trimmed, "x1"), 0.0f);
            const float y1 = firstNumber(attr(trimmed, "y1"), 0.0f);
            const float x2 = firstNumber(attr(trimmed, "x2"), 0.0f);
            const float y2 = firstNumber(attr(trimmed, "y2"), 0.0f);
            std::vector<glm::vec2> pts = {xf.apply({x1, y1}), xf.apply({x2, y2})};
            pushContour(std::move(pts), false, rgb);
            continue;
        }
    }

    if (out.contours.empty()) {
        error = "no drawable shapes (path/rect/circle/ellipse/polygon/polyline/line)";
        return false;
    }
    return true;
}
