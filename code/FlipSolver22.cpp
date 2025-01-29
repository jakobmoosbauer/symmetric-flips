// Fast matrix multiplication search algorthim, written by Mike Poole.
// Flip graph method fast solver for version 22 - September 2024.
// Copyright (C) Mike Poole, September 2024.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <iostream>
#include <fstream>
#include <vector>
#include <random>

typedef unsigned long long int vlong;

// Bespoke dictionary data structure class for flip graph.
class fgdict {
public:
    unsigned int lasthash;
    int* count;
    vlong* key;
    int* value;

    // Constructor.
    fgdict() {
        lasthash = 0;
        count = new int[1048576];
        key = new vlong[1048576];
        value = new int[1048576];
        for (int i = 0; i < 65536; i++) {
            count[i << 4] = 0;
        }
    }

    // Destructor.
    ~fgdict() {
        delete[] count, key, value;
    }

    // Calculate size of dictionary.
    int size() {
        int l = 0;
        for (int i = 0; i < 65536; i++) {
            l += count[i << 4];
        }
        return l;
    }

    // Hash function.
    unsigned int hash(vlong k) {
        unsigned int h = (k % 65213) << 4;
        return h;
    }

    // Check if dictionary contains key.
    int contains(vlong k) {
        lasthash = hash(k);
        int c = count[lasthash];
        if (c == 0) {
            return 0;
        }
        else if (c == 1) {
            if (key[lasthash] == k) {
                return 1;
            }
            else {
                return 0;
            }
        }
        else {
            for (int i = c - 1; i >= 0; i--) {
                if (key[lasthash + i] == k) {
                    return 1;
                }
            }
            return 0;
        }
    }

    // Add key/value pair, assumes key not already present.
    void add(vlong k, int v) {
        lasthash = hash(k);
        int b = lasthash + count[lasthash];
        key[b] = k;
        value[b] = v;
        count[lasthash]++;
    }

    // Add key/value pair, assumes key not already present, hash already calculated.
    void addx(vlong k, int v) {
        int b = lasthash + count[lasthash];
        key[b] = k;
        value[b] = v;
        count[lasthash]++;
    }

    // Remove item, assumes key exists.
    void remove(vlong k) {
        lasthash = hash(k);
        int c = count[lasthash];
        if (c == 1) {
            count[lasthash] = 0;
        }
        else {
            int i = lasthash + c - 1;
            vlong x = key[i];
            int v = value[i];
            while (x != k) {
                i--;
                vlong y = x;
                x = key[i];
                key[i] = y;
                int w = v;
                v = value[i];
                value[i] = w;
            }
            count[lasthash]--;
        }
    }

    // Remove item, assumes key exists, hash already calculated.
    void removex(vlong k) {
        int c = count[lasthash];
        if (c == 1) {
            count[lasthash] = 0;
        }
        else {
            int i = lasthash + c - 1;
            vlong x = key[i];
            int v = value[i];
            while (x != k) {
                i--;
                vlong y = x;
                x = key[i];
                key[i] = y;
                int w = v;
                v = value[i];
                value[i] = w;
            }
            count[lasthash]--;
        }
    }

    // Replace value for specified key, assumes key exists.
    void replace(vlong k, int v) {
        lasthash = hash(k);
        int c = count[lasthash];
        if (c == 1) {
            value[lasthash] = v;
        }
        else {
            int i = lasthash + c - 1;
            vlong x = key[i];
            while (x != k) {
                i--;
                x = key[i];
            }
            value[i] = v;
        }
    }

    // Replace value for specified key, assumes key exists, hash already calculated.
    void replacex(vlong k, int v) {
        int c = count[lasthash];
        if (c == 1) {
            value[lasthash] = v;
        }
        else {
            int i = lasthash + c - 1;
            vlong x = key[i];
            while (x != k) {
                i--;
                x = key[i];
            }
            value[i] = v;
        }
    }

    // Get value for specified key, assumes key exists.
    int getvalue(vlong k) {
        lasthash = hash(k);
        int c = count[lasthash];
        if (c == 1) {
            return value[lasthash];
        }
        else {
            int i = lasthash + c - 1;
            vlong x = key[i];
            while (x != k) {
                i--;
                x = key[i];
            }
            return value[i];
        }
    }

    // Get value for specified key, assumes key exists, hash already calculated.
    int getvaluex(vlong k) {
        int c = count[lasthash];
        if (c == 1) {
            return value[lasthash];
        }
        else {
            int i = lasthash + c - 1;
            vlong x = key[i];
            while (x != k) {
                i--;
                x = key[i];
            }
            return value[i];
        }
    }
};

// Bookkeeping associated with deleting a multiplication component.
inline void flipdel(int unarray[], std::vector<int>& avail, int nomuls,
    fgdict& uniques,
    fgdict& twoplusd,
    std::vector<vlong>& twoplusl, int r, vlong v) {
    int b = uniques.getvalue(v);
    int l = unarray[b];
    if (l == 2) {
        twoplusd.lasthash = uniques.lasthash;
        int rsi = twoplusd.getvaluex(v);
        vlong rsl = twoplusl.back();
        twoplusd.replace(rsl, rsi);
        twoplusl[rsi] = rsl;
        twoplusl.pop_back();
        twoplusd.lasthash = uniques.lasthash;
        twoplusd.removex(v);
    }
    if (l == 1) {
        avail.push_back(b);
        uniques.removex(v);
    }
    else {
        int i = b + l;
        int x = unarray[i];
        while (x != r) {
            i--;
            int y = x;
            x = unarray[i];
            unarray[i] = y;
        }
        unarray[b] = l - 1;
    }
}

// Bookkeeping associated with adding a multiplication component.
inline void flipadd(int unarray[], std::vector<int>& avail, int nomuls,
    fgdict& uniques,
    fgdict& twoplusd,
    std::vector<vlong>& twoplusl, int r, vlong v) {
    int ct = uniques.contains(v);
    if (ct) {
        int b = uniques.getvaluex(v);
        int l = unarray[b];
        if (l == 1) {
            twoplusd.lasthash = uniques.lasthash;
            twoplusd.addx(v, twoplusl.size());
            twoplusl.push_back(v);
        }
        l++;
        unarray[b + l] = r;
        unarray[b] = l;
    }
    else {
        int b = avail.back();
        avail.pop_back();
        uniques.addx(v, b);
        unarray[b + 1] = r;
        unarray[b] = 1;
    }
}

// Returns number of set bits.
inline int bitcount(vlong var) {
    int c = 0;
    vlong n = var;
    while (n) {
        c++;
        n &= (n - 1);
    }
    return c;
}

// Returns non-zero (true) if number of set bits < exceed, else zero (false).
inline int bitlimit(vlong var, int exceed) {
    int m = exceed;
    vlong n = var;
    while (n && m) {
        m--;
        n &= (n - 1);
    }
    return m;
}

// Returns updated flip limit on new overall rank reduction.
vlong updatelimit(vlong limit, vlong flips, int termination, int split, int achieved, int target, int symm, vlong flimit) {
    vlong rlimit;
    if (termination == 0) {
        rlimit = flimit;
    }
    else if (termination == 1) {
        int steps = (achieved - target) / symm;
        rlimit = flips + (flimit - flips) / steps;
    }
    else if (termination == 2) {
        rlimit = flips + flimit;
    }
    else {
        vlong slimit = split * flimit / 100;
        if (achieved > termination) {
            int steps = (achieved - termination) / symm;
            rlimit = flips + (slimit - flips) / steps;
        }
        else {
            int steps = (achieved - target) / symm;
            rlimit = flips + (flimit - flips) / steps;
        }
    }
    return rlimit;
}

// C++ implementation of original Python solver function.
int main(int argc, char* argv[]) {

    std::ifstream input_file(argv[1]);
    vlong flips, flimit, plimit;
    int nomuls, rcode, target, rseed, symm, achieved, maxplus, minmuls, maxsize, termination, split;
    input_file >> nomuls >> flips >> rcode >> target >> flimit >> plimit >> termination >> rseed >> symm >> maxplus >> split >> minmuls >> maxsize;

    std::vector<vlong> muls;
    std::vector<vlong> best;
    for (int i = 0; i < nomuls; i++) {
        vlong m;
        input_file >> m;
        muls.push_back(m);
        best.push_back(m);
    }

    std::vector<int> me(nomuls, 0);
    std::vector<int> mf(nomuls, 0);
    for (int i = 0; i < nomuls; i += 3) {
        me[i] = i + 2;
        mf[i] = i + 1;
        me[i + 1] = i;
        mf[i + 1] = i + 2;
        me[i + 2] = i + 1;
        mf[i + 2] = i;
    }

    std::mt19937 mt(rseed);

    fgdict uniques;
    int* unarray = new int[nomuls * (nomuls + 1)];
    std::vector<int> avail;
    for (int i = 0; i < nomuls; i++) {
        int b = i * (nomuls + 1);
        avail.push_back(b);
    }
    fgdict twoplusd;
    std::vector<vlong> twoplusl;

    std::vector<std::vector<int>> permit;
    permit.reserve(nomuls);
    for (int i = 0; i < nomuls; i++) {
        std::vector<int> p;
        p.reserve(nomuls);
        for (int j = 0; j < nomuls; j++) {
            if (i / symm == j / symm) {
                p.push_back(0);
            }
            else {
                p.push_back(1);
            }
        }
        permit.push_back(p);
    }

    achieved = 0;
    for (int i = 0; i < nomuls; i++) {
        vlong m = muls[i];
        if (m > 0) {
            if (uniques.contains(m)) {
                int b = uniques.getvalue(m);
                int l = unarray[b];
                l++;
                unarray[b + l] = i;
                unarray[b] = l;
                if (!twoplusd.contains(m)) {
                    twoplusd.add(m, twoplusl.size());
                    twoplusl.push_back(m);
                }
            }
            else {
                int b = avail.back();
                avail.pop_back();
                uniques.add(m, b);
                unarray[b] = 1;
                unarray[b + 1] = i;
            }
            achieved += 1;
        }
    }

    std::vector<int> combs;
    combs.reserve(100);
    combs.push_back(0);
    combs.push_back(0);
    std::vector<int> ps;
    ps.reserve(6400);
    std::vector<int> qs;
    qs.reserve(6400);
    for (int x = 1; x < 80; x++) {
        for (int y = 0; y < x; y++) {
            ps.push_back(x);
            qs.push_back(y);
            ps.push_back(y);
            qs.push_back(x);
        }
        combs.push_back(ps.size());
    }

    vlong plus = 0;
    rcode = 0;
    int exceed = 1 - maxsize;
    vlong plusby;
    if (achieved >= maxplus) {
        plusby = flimit * 1007;
    }
    else if (plimit < 0) {
        plusby = flips + symm + mt() % (-2 * plimit);
    }
    else {
        plusby = flips + plimit;
    }
    vlong recovery = 5000000000;
    minmuls = achieved;
    vlong limit = 0;
    limit = updatelimit(limit, flips, termination, split, achieved, target, symm, flimit);

    if (symm == 3) {
        while (true) {
            flips += 3;

            int p, q;
            vlong mpe, mpf, mqe, mqf, mpen, mqfn;
            if (maxsize == 0) {
                while (true) {
                    unsigned int sample = mt();
                    vlong v = twoplusl[sample % twoplusl.size()];
                    int b = uniques.getvalue(v);
                    int l = unarray[b];
                    b++;
                    if (l == 2) {
                        if (sample & 65536) {
                            p = unarray[b];
                            q = unarray[b + 1];
                        }
                        else {
                            p = unarray[b + 1];
                            q = unarray[b];
                        }
                    }
                    else {
                        int x = (sample >> 16) % combs[l];
                        p = unarray[b + ps[x]];
                        q = unarray[b + qs[x]];
                    }
                    if (permit[p][q]) {
                        break;
                    }
                }
                mpe = muls[me[p]];
                mpf = muls[mf[p]];
                mqe = muls[me[q]];
                mqf = muls[mf[q]];
                mpen = mqe ^ mpe;
                mqfn = mqf ^ mpf;
            }
            else if (maxsize > 0) {
                int k;
                for (k = 0; k < 1000; k++) {
                    unsigned int sample = mt();
                    vlong v = twoplusl[sample % twoplusl.size()];
                    int b = uniques.getvalue(v);
                    int l = unarray[b];
                    b++;
                    if (l == 2) {
                        if (sample & 65536) {
                            p = unarray[b];
                            q = unarray[b + 1];
                        }
                        else {
                            p = unarray[b + 1];
                            q = unarray[b];
                        }
                    }
                    else {
                        int x = (sample >> 16) % combs[l];
                        p = unarray[b + ps[x]];
                        q = unarray[b + qs[x]];
                    }
                    mpe = muls[me[p]];
                    mpf = muls[mf[p]];
                    mqe = muls[me[q]];
                    mqf = muls[mf[q]];
                    mpen = mqe ^ mpe;
                    mqfn = mqf ^ mpf;
                    int psize = bitcount(muls[p]) * bitcount(mpen) * bitcount(mpf);
                    int qsize = bitcount(muls[q]) * bitcount(mqe) * bitcount(mqfn);
                    if (permit[p][q] && psize <= maxsize && qsize <= maxsize) {
                        break;
                    }
                }
                if (k == 1000) {
                    rcode = 6;
                    break;
                }
            }
            else {
                int k;
                for (k = 0; k < 1000; k++) {
                    unsigned int sample = mt();
                    vlong v = twoplusl[sample % twoplusl.size()];
                    int b = uniques.getvalue(v);
                    int l = unarray[b];
                    b++;
                    if (l == 2) {
                        if (sample & 65536) {
                            p = unarray[b];
                            q = unarray[b + 1];
                        }
                        else {
                            p = unarray[b + 1];
                            q = unarray[b];
                        }
                    }
                    else {
                        int x = (sample >> 16) % combs[l];
                        p = unarray[b + ps[x]];
                        q = unarray[b + qs[x]];
                    }
                    mpe = muls[me[p]];
                    mpf = muls[mf[p]];
                    mqe = muls[me[q]];
                    mqf = muls[mf[q]];
                    mpen = mqe ^ mpe;
                    mqfn = mqf ^ mpf;
                    if (permit[p][q] && bitlimit(mpen, exceed) && bitlimit(mqfn, exceed)) {
                        break;
                    }
                }
                if (k == 1000) {
                    rcode = 6;
                    break;
                }
            }
            flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpe);
            flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpen);
            muls[me[p]] = mpen;

            flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqf);
            flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqfn);
            muls[mf[q]] = mqfn;

            if (mpen == 0) {
                vlong mpd = muls[p];
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, p, mpd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpen);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[p], mpf);
                muls[p] = 0;
                muls[mf[p]] = 0;
                achieved -= 3;
                if (achieved < minmuls) {
                    minmuls = achieved;
                    if (achieved > target) {
                        limit = updatelimit(limit, flips, termination, split, achieved, target, symm, flimit);
                    }
                }
                if (achieved <= minmuls) {
                    for (int l = 0; l < nomuls; l++) {
                        best[l] = muls[l];
                    }
                }
                if (achieved >= maxplus) {
                    plusby = flimit * 1007;
                }
                else if (plimit < 0) {
                    plusby = flips + symm + mt() % (-2 * plimit);
                }
                else {
                    plusby = flips + plimit;
                }
                if (twoplusl.size() == 0) {
                    rcode = -1;
                    break;
                }
                if (achieved <= target) {
                    break;
                }
                bool trigger = true;
                for (int j = 0; j < twoplusl.size(); j++) {
                    vlong v = twoplusl[j];
                    int b = uniques.getvalue(v);
                    int t = unarray[b + 1] / 3;
                    for (int i = 1; i < unarray[b]; i++) {
                        int u = unarray[b + i + 1] / 3;
                        if (t != u) {
                            trigger = false;
                        }
                    }
                }
                if (trigger) {
                    plusby = flips;
                }
            }

            if (mqfn == 0) {
                vlong mqd = muls[q];
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, q, mqd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[q], mqe);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqfn);
                muls[q] = 0;
                muls[me[q]] = 0;
                achieved -= 3;
                if (achieved < minmuls) {
                    minmuls = achieved;
                    if (achieved > target) {
                        limit = updatelimit(limit, flips, termination, split, achieved, target, symm, flimit);
                    }
                }
                if (achieved <= minmuls) {
                    for (int l = 0; l < nomuls; l++) {
                        best[l] = muls[l];
                    }
                }
                if (achieved >= maxplus) {
                    plusby = flimit * 1007;
                }
                else if (plimit < 0) {
                    plusby = flips + symm + mt() % (-2 * plimit);
                }
                else {
                    plusby = flips + plimit;
                }
                if (twoplusl.size() == 0) {
                    rcode = -1;
                    break;
                }
                if (achieved <= target) {
                    break;
                }
                bool trigger = true;
                for (int j = 0; j < twoplusl.size(); j++) {
                    vlong v = twoplusl[j];
                    int b = uniques.getvalue(v);
                    int t = unarray[b + 1] / 3;
                    for (int i = 1; i < unarray[b]; i++) {
                        int u = unarray[b + i + 1] / 3;
                        if (t != u) {
                            trigger = false;
                        }
                    }
                }
                if (trigger) {
                    plusby = flips;
                }
            }

            if (flips >= plusby) {
                if (flips >= recovery) {
                    recovery += 5000000000;
                    std::ofstream output_file(argv[1]);
                    output_file << nomuls << " " << flips << " " << 2 << " " << target << " " << flimit << " ";
                    output_file << plimit << " " << termination << " " << rseed << " " << symm << " " << maxplus << " ";
                    output_file << achieved << " " << minmuls << " " << plus << "\n";
                    for (vlong m : muls) {
                        output_file << m << "\n";
                    }
                }
                int r;
                for (r = 0; r < nomuls; r++) {
                    if (muls[r] == 0) break;
                }
                int p, q;
                vlong mpd, mpe, mpf, mqd, mqe, mqf;
                vlong mpdn, mpen, mpfn, mqdn, mqen, mqfn, mrdn, mren, mrfn;
                while (true) {
                    p = mt() % nomuls;
                    q = mt() % nomuls;
                    mpd = muls[p];
                    mpe = muls[me[p]];
                    mpf = muls[mf[p]];
                    mqd = muls[q];
                    mqe = muls[me[q]];
                    mqf = muls[mf[q]];
                    mpdn = mpd;
                    mpen = mpe ^ mqe;
                    mpfn = mpf;
                    mqdn = mpd;
                    mqen = mqe;
                    mqfn = mpf ^ mqf;
                    mrdn = mpd ^ mqd;
                    mren = mqe;
                    mrfn = mqf;
                    bool ok = true;
                    if (maxsize > 0) {
                        int psize = bitcount(mpdn) * bitcount(mpen) * bitcount(mpfn);
                        int qsize = bitcount(mqdn) * bitcount(mqen) * bitcount(mqfn);
                        int rsize = bitcount(mrdn) * bitcount(mren) * bitcount(mrfn);
                        if (psize > maxsize || qsize > maxsize || rsize > maxsize) {
                            ok = false;
                        }
                    }
                    else if (maxsize < 0) {
                        if (!(bitlimit(mpen, exceed) && bitlimit(mqfn, exceed) && bitlimit(mrdn, exceed))) {
                            ok = false;
                        }
                    }
                    if (mpd == 0 || mqd == 0) ok = false;
                    if (mpd == mqd || mpe == mqe || mpf == mqf) ok = false;
                    if (!permit[p][q]) ok = false;
                    if (ok) break;
                }
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpe);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpen);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, q, mqd);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, q, mpd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqf);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqfn);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, r, mrdn);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[r], mqe);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[r], mqf);
                muls[p] = mpdn;
                muls[me[p]] = mpen;
                muls[mf[p]] = mpfn;
                muls[q] = mqdn;
                muls[me[q]] = mqen;
                muls[mf[q]] = mqfn;
                muls[r] = mrdn;
                muls[me[r]] = mren;
                muls[mf[r]] = mrfn;
                plus += 3;
                achieved += 3;
                if (achieved >= maxplus) {
                    plusby = flimit * 1007;
                }
                else if (plimit < 0) {
                    plusby = flips + symm + mt() % (-2 * plimit);
                }
                else {
                    plusby = flips + plimit;
                }
            }

            if (flips >= limit) {
                if (flips >= flimit) {
                    rcode = 1;
                }
                else {
                    rcode = 2;
                }
                break;
            }
        }
    }

    if (symm == 6) {
        while (true) {
            flips += 6;

            int p, q;
            vlong mpd, mpe, mpf, mqd, mqe, mqf, mpen, mqfn;
            if (maxsize == 0) {
                while (true) {
                    unsigned int sample = mt();
                    vlong v = twoplusl[sample % twoplusl.size()];
                    int b = uniques.getvalue(v);
                    int l = unarray[b];
                    b++;
                    if (l == 2) {
                        if (sample & 65536) {
                            p = unarray[b];
                            q = unarray[b + 1];
                        }
                        else {
                            p = unarray[b + 1];
                            q = unarray[b];
                        }
                    }
                    else {
                        int x = (sample >> 16) % combs[l];
                        p = unarray[b + ps[x]];
                        q = unarray[b + qs[x]];
                    }
                    if (permit[p][q]) {
                        break;
                    }
                }
                mpd = muls[p];
                mpe = muls[me[p]];
                mpf = muls[mf[p]];
                mqd = muls[q];
                mqe = muls[me[q]];
                mqf = muls[mf[q]];
                mpen = mqe ^ mpe;
                mqfn = mqf ^ mpf;
            }
            else if (maxsize > 0) {
                int k;
                for (k = 0; k < 1000; k++) {
                    unsigned int sample = mt();
                    vlong v = twoplusl[sample % twoplusl.size()];
                    int b = uniques.getvalue(v);
                    int l = unarray[b];
                    b++;
                    if (l == 2) {
                        if (sample & 65536) {
                            p = unarray[b];
                            q = unarray[b + 1];
                        }
                        else {
                            p = unarray[b + 1];
                            q = unarray[b];
                        }
                    }
                    else {
                        int x = (sample >> 16) % combs[l];
                        p = unarray[b + ps[x]];
                        q = unarray[b + qs[x]];
                    }
                    mpd = muls[p];
                    mpe = muls[me[p]];
                    mpf = muls[mf[p]];
                    mqd = muls[q];
                    mqe = muls[me[q]];
                    mqf = muls[mf[q]];
                    mpen = mqe ^ mpe;
                    mqfn = mqf ^ mpf;
                    int psize = bitcount(mpd) * bitcount(mpen) * bitcount(mpf);
                    int qsize = bitcount(mqd) * bitcount(mqe) * bitcount(mqfn);
                    if (permit[p][q] && psize <= maxsize && qsize <= maxsize) {
                        break;
                    }
                }
                if (k == 1000) {
                    rcode = 6;
                    break;
                }
            }
            else {
                int k;
                for (k = 0; k < 1000; k++) {
                    unsigned int sample = mt();
                    vlong v = twoplusl[sample % twoplusl.size()];
                    int b = uniques.getvalue(v);
                    int l = unarray[b];
                    b++;
                    if (l == 2) {
                        if (sample & 65536) {
                            p = unarray[b];
                            q = unarray[b + 1];
                        }
                        else {
                            p = unarray[b + 1];
                            q = unarray[b];
                        }
                    }
                    else {
                        int x = (sample >> 16) % combs[l];
                        p = unarray[b + ps[x]];
                        q = unarray[b + qs[x]];
                    }
                    mpd = muls[p];
                    mpe = muls[me[p]];
                    mpf = muls[mf[p]];
                    mqd = muls[q];
                    mqe = muls[me[q]];
                    mqf = muls[mf[q]];
                    mpen = mqe ^ mpe;
                    mqfn = mqf ^ mpf;
                    if (permit[p][q] && bitlimit(mpen, exceed) && bitlimit(mqfn, exceed)) {
                        break;
                    }
                }
                if (k == 1000) {
                    rcode = 6;
                    break;
                }
            }

            int x = p % 6;
            int pp;
            if (x < 3) {
                pp = p + 3;
            }
            else {
                pp = p - 3;
            }
            x = q % 6;
            int qq;
            if (x < 3) {
                qq = q + 3;
            }
            else {
                qq = q - 3;
            }

            vlong mppd = muls[pp];
            vlong mppe = muls[me[pp]];
            vlong mppf = muls[mf[pp]];
            vlong mqqd = muls[qq];
            vlong mqqe = muls[me[qq]];
            vlong mqqf = muls[mf[qq]];
            vlong mppen = mqqe ^ mppe;
            vlong mqqfn = mqqf ^ mppf;

            flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpe);
            flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpen);
            muls[me[p]] = mpen;
            flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[pp], mppe);
            flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[pp], mppen);
            muls[me[pp]] = mppen;

            flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqf);
            flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqfn);
            muls[mf[q]] = mqfn;
            flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[qq], mqqf);
            flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[qq], mqqfn);
            muls[mf[qq]] = mqqfn;

            if (mpen == 0 || (mpd == mppd && mpen == mppen && mpf == mppf)) {
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, p, mpd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpen);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[p], mpf);
                muls[p] = 0;
                muls[mf[p]] = 0;
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, pp, mppd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[pp], mppen);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[pp], mppf);
                muls[pp] = 0;
                muls[mf[pp]] = 0;
                if (mpen != 0) {
                    muls[me[p]] = 0;
                    muls[me[pp]] = 0;
                }
                achieved -= 6;
                if (achieved < minmuls) {
                    minmuls = achieved;
                    if (achieved > target) {
                        limit = updatelimit(limit, flips, termination, split, achieved, target, symm, flimit);
                    }
                }
                if (achieved <= minmuls) {
                    for (int l = 0; l < nomuls; l++) {
                        best[l] = muls[l];
                    }
                }
                if (achieved >= maxplus) {
                    plusby = flimit * 1007;
                }
                else if (plimit < 0) {
                    plusby = flips + symm + mt() % (-2 * plimit);
                }
                else {
                    plusby = flips + plimit;
                }
                if (twoplusl.size() == 0) {
                    rcode = -1;
                    break;
                }
                if (achieved <= target) {
                    break;
                }
                bool trigger = true;
                for (int j = 0; j < twoplusl.size(); j++) {
                    vlong v = twoplusl[j];
                    int b = uniques.getvalue(v);
                    int t = unarray[b + 1] / 6;
                    for (int i = 1; i < unarray[b]; i++) {
                        int u = unarray[b + i + 1] / 6;
                        if (t != u) {
                            trigger = false;
                        }
                    }
                }
                if (trigger) {
                    plusby = flips;
                }
            }

            if (mqfn == 0 || (mqd == mqqd && mqe == mqqe && mqfn == mqqfn)) {
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, q, mqd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[q], mqe);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqfn);
                muls[q] = 0;
                muls[me[q]] = 0;
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, qq, mqqd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[qq], mqqe);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[qq], mqqfn);
                muls[qq] = 0;
                muls[me[qq]] = 0;
                if (mqfn != 0) {
                    muls[mf[q]] = 0;
                    muls[mf[qq]] = 0;
                }
                achieved -= 6;
                if (achieved < minmuls) {
                    minmuls = achieved;
                    if (achieved > target) {
                        limit = updatelimit(limit, flips, termination, split, achieved, target, symm, flimit);
                    }
                }
                if (achieved <= minmuls) {
                    for (int l = 0; l < nomuls; l++) {
                        best[l] = muls[l];
                    }
                }
                if (achieved >= maxplus) {
                    plusby = flimit * 1007;
                }
                else if (plimit < 0) {
                    plusby = flips + symm + mt() % (-2 * plimit);
                }
                else {
                    plusby = flips + plimit;
                }
                if (twoplusl.size() == 0) {
                    rcode = -1;
                    break;
                }
                if (achieved <= target) {
                    break;
                }
                bool trigger = true;
                for (int j = 0; j < twoplusl.size(); j++) {
                    vlong v = twoplusl[j];
                    int b = uniques.getvalue(v);
                    int t = unarray[b + 1] / 6;
                    for (int i = 1; i < unarray[b]; i++) {
                        int u = unarray[b + i + 1] / 6;
                        if (t != u) {
                            trigger = false;
                        }
                    }
                }
                if (trigger) {
                    plusby = flips;
                }
            }

            if (flips >= plusby) {
                if (flips >= recovery) {
                    recovery += 5000000000;
                    std::ofstream output_file(argv[1]);
                    output_file << nomuls << " " << flips << " " << 2 << " " << target << " " << flimit << " ";
                    output_file << plimit << " " << termination << " " << rseed << " " << symm << " " << maxplus << " ";
                    output_file << achieved << " " << minmuls << " " << plus << "\n";
                    for (vlong m : muls) {
                        output_file << m << "\n";
                    }
                }
                int r;
                for (r = 0; r < nomuls; r++) {
                    if (muls[r] == 0) break;
                }
                int rr = r + 3;
                int p, q, pp, qq;
                vlong mpd, mpe, mpf, mqd, mqe, mqf;
                vlong mpdn, mpen, mpfn, mqdn, mqen, mqfn, mrdn, mren, mrfn;
                vlong mppd, mppe, mppf, mqqd, mqqe, mqqf;
                vlong mppdn, mppen, mppfn, mqqdn, mqqen, mqqfn, mrrdn, mrren, mrrfn;
                while (true) {
                    p = mt() % nomuls;
                    q = mt() % nomuls;
                    int x = p % 6;
                    if (x < 3) {
                        pp = p + 3;
                    }
                    else {
                        pp = p - 3;
                    }
                    x = q % 6;
                    if (x < 3) {
                        qq = q + 3;
                    }
                    else {
                        qq = q - 3;
                    }
                    mpd = muls[p];
                    mpe = muls[me[p]];
                    mpf = muls[mf[p]];
                    mqd = muls[q];
                    mqe = muls[me[q]];
                    mqf = muls[mf[q]];
                    mpdn = mpd;
                    mpen = mpe ^ mqe;
                    mpfn = mpf;
                    mqdn = mpd;
                    mqen = mqe;
                    mqfn = mpf ^ mqf;
                    mrdn = mpd ^ mqd;
                    mren = mqe;
                    mrfn = mqf;
                    mppd = muls[pp];
                    mppe = muls[me[pp]];
                    mppf = muls[mf[pp]];
                    mqqd = muls[qq];
                    mqqe = muls[me[qq]];
                    mqqf = muls[mf[qq]];
                    mppdn = mppd;
                    mppen = mppe ^ mqqe;
                    mppfn = mppf;
                    mqqdn = mppd;
                    mqqen = mqqe;
                    mqqfn = mppf ^ mqqf;
                    mrrdn = mppd ^ mqqd;
                    mrren = mqqe;
                    mrrfn = mqqf;
                    bool ok = true;
                    if (maxsize > 0) {
                        int psize = bitcount(mpdn) * bitcount(mpen) * bitcount(mpfn);
                        int qsize = bitcount(mqdn) * bitcount(mqen) * bitcount(mqfn);
                        int rsize = bitcount(mrdn) * bitcount(mren) * bitcount(mrfn);
                        if (psize > maxsize || qsize > maxsize || rsize > maxsize) {
                            ok = false;
                        }
                    }
                    else if (maxsize < 0) {
                        if (!(bitlimit(mpen, exceed) && bitlimit(mqfn, exceed) && bitlimit(mrdn, exceed))) {
                            ok = false;
                        }
                    }
                    if (mpd == 0 || mqd == 0) ok = false;
                    if (mppd == 0 || mqqd == 0) ok = false;
                    if (mpd == mqd || mpe == mqe || mpf == mqf) ok = false;
                    if (mppd == mqqd || mppe == mqqe || mppf == mqqf) ok = false;
                    if (!permit[p][q]) ok = false;
                    if (ok) break;
                }
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpe);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[p], mpen);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, q, mqd);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, q, mpd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqf);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[q], mqfn);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, r, mrdn);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[r], mqe);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[r], mqf);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[pp], mppe);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[pp], mppen);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, qq, mqqd);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, qq, mppd);
                flipdel(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[qq], mqqf);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[qq], mqqfn);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, rr, mrrdn);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, me[rr], mqqe);
                flipadd(unarray, avail, nomuls, uniques, twoplusd, twoplusl, mf[rr], mqqf);
                muls[p] = mpdn;
                muls[me[p]] = mpen;
                muls[mf[p]] = mpfn;
                muls[q] = mqdn;
                muls[me[q]] = mqen;
                muls[mf[q]] = mqfn;
                muls[r] = mrdn;
                muls[me[r]] = mren;
                muls[mf[r]] = mrfn;
                muls[pp] = mppdn;
                muls[me[pp]] = mppen;
                muls[mf[pp]] = mppfn;
                muls[qq] = mqqdn;
                muls[me[qq]] = mqqen;
                muls[mf[qq]] = mqqfn;
                muls[rr] = mrrdn;
                muls[me[rr]] = mrren;
                muls[mf[rr]] = mrrfn;
                plus += 6;
                achieved += 6;
                if (achieved >= maxplus) {
                    plusby = flimit * 1007;
                }
                else if (plimit < 0) {
                    plusby = flips + symm + mt() % (-2 * plimit);
                }
                else {
                    plusby = flips + plimit;
                }
            }

            if (flips >= limit) {
                if (flips >= flimit) {
                    rcode = 1;
                }
                else {
                    rcode = 2;
                }
                break;
            }
        }
    }

    std::ofstream output_file(argv[1]);
    output_file << nomuls << " " << flips << " " << rcode << " " << target << " " << flimit << " ";
    output_file << plimit << " " << termination << " " << rseed << " " << symm << " " << maxplus << " ";
    output_file << achieved << " " << minmuls << " " << plus << "\n";
    if (minmuls < achieved) {
        for (vlong m : best) {
            output_file << m << "\n";
        }
    }
    else {
        for (vlong m : muls) {
            output_file << m << "\n";
        }
    }

    delete[] unarray;

    return 0;
}
