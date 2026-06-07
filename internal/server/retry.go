package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pbPkg "github.com/a876691666/deepseek-cursor-proxy/internal/pocketbase"
)

// doUpstreamRequest performs an HTTP request with retry logic.
// It retries on network errors and 5xx server errors up to Config.RetryCount times.
// bodyBytes is saved for re-creating the request body on each retry attempt.
// Returns (response, nil) on success, or (nil, error) after all retries are exhausted.
func (s *Server) doUpstreamRequest(req *http.Request, bodyBytes []byte) (*http.Response, error) {
	maxRetries := s.Config.RetryCount
	if maxRetries <= 0 {
		return s.Client.Do(req)
	}

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			backoff := time.Duration(attempt) * time.Second
			s.Logger.Printf("retrying upstream request attempt=%d/%d backoff=%v", attempt+1, maxRetries+1, backoff)
			time.Sleep(backoff)
		}

		resp, err := s.Client.Do(req)
		if err != nil {
			lastErr = err
			s.Logger.Printf("upstream request failed attempt=%d/%d error=%s", attempt+1, maxRetries+1, err)
			continue
		}

		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("upstream returned status %d", resp.StatusCode)
			resp.Body.Close()
			s.Logger.Printf("upstream request returned %d attempt=%d/%d", resp.StatusCode, attempt+1, maxRetries+1)
			continue
		}

		return resp, nil
	}

	return nil, lastErr
}

// logUpstreamError records upstream request/response error details to the PocketBase error_log collection.
// r is the original client request, upstreamReq is the upstream HTTP request being sent,
// upstreamBody is the request body bytes, resp is the upstream response (nil for network errors),
// respBody is the pre-read response body string (empty if not available),
// errMsg is the error description.
func (s *Server) logUpstreamError(r *http.Request, upstreamReq *http.Request, upstreamBody []byte, resp *http.Response, respBody, errMsg string) {
	if s.PB == nil {
		return
	}

	// Capture request headers
	reqHeaders, _ := json.Marshal(upstreamReq.Header)

	// Capture response headers if available
	var respHeaders string
	if resp != nil {
		respHeadersMap := make(map[string]string)
		for k := range resp.Header {
			respHeadersMap[k] = resp.Header.Get(k)
		}
		if b, err := json.Marshal(respHeadersMap); err == nil {
			respHeaders = string(b)
		}
	}

	statusCode := 0
	if resp != nil {
		statusCode = resp.StatusCode
	}

	record := pbPkg.ErrorLogRecord{
		Endpoint:        r.URL.Path,
		Method:          r.Method,
		StatusCode:      statusCode,
		RequestHeaders:  truncateString(string(reqHeaders), 5000),
		RequestBody:     truncateString(string(upstreamBody), 10000),
		RequestQuery:    r.URL.RawQuery,
		ResponseHeaders: truncateString(respHeaders, 5000),
		ResponseBody:    truncateString(respBody, 10000),
		ErrorMessage:    truncateString(errMsg, 2000),
	}

	if err := pbPkg.RecordErrorLog(s.PB, record); err != nil {
		s.Logger.Printf("failed to record error log: %s", err)
	}
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen]
}
